# -*- coding: utf-8 -*-
"""
project.py
Objetivo: Buscar prontu?rios em m?ltiplos bancos (por "posto") via SQLs externos
em ./sql/<POSTO>.sql, aplicar filtros case/acento-insens?veis, consolidar resultados,
persistir um JSON em ./json e, opcionalmente, enviar um resumo para a Groq para obter
uma sugest?o. Ao final, perguntar se deve apagar o JSON.

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv groq
  ODBC Driver 17/18 for SQL Server instalado no Windows

Configura??o:
  Vari?veis de ambiente no .env (na raiz do projeto), ex.:
    DB_HOST_A, DB_PORT_A, DB_BASE_A, DB_USER_A, DB_PASSWORD_A
    DB_ENCRYPT=yes|no, DB_TRUST_CERT=yes|no, DB_TIMEOUT=5
  Postos n?o configurados (sem HOST/BASE) s?o automaticamente ignorados.
"""

# =========================
# IMPORTS E CAMINHOS
# =========================
import os
import json
import uuid
import hashlib
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Diret?rios base do projeto, SQLs e JSONs
BASE_DIR = os.path.dirname(__file__)
SQL_DIR = os.path.join(BASE_DIR, "sql")
JSON_DIR = os.path.join(BASE_DIR, "json")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
DEFAULT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "prompt_analise_atds_anteriores.txt")

# Carrega vari?veis do .env
load_dotenv(os.path.join(BASE_DIR, ".env"))

# =========================
# FUN??ES DE SUPORTE
# =========================
def ensure_dirs():
    os.makedirs(SQL_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(PROMPTS_DIR, exist_ok=True)


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else v


# Postos suportados. Ajuste conforme ambiente real.
POSTOS = ["A", "B", "C", "D", "G", "I", "J", "M"]


def build_conn_str(
    host: str,
    base: str,
    user: str,
    pwd: str,
    port: str,
    encrypt: str,
    trust_cert: str,
    timeout: str,
) -> str:
    server = f"tcp:{host},{port or '1433'}"
    common = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};DATABASE={base};"
        f"Encrypt={encrypt};TrustServerCertificate={trust_cert};"
        f"Connection Timeout={timeout or '5'}"
    )
    if user:
        return f"{common};UID={user};PWD={pwd}"
    else:
        return f"{common};Trusted_Connection=yes"


def build_conns_from_env():
    encrypt = _env("DB_ENCRYPT", "yes")
    trust_cert = _env("DB_TRUST_CERT", "yes")
    timeout = _env("DB_TIMEOUT", "5")

    conns = {}
    for p in POSTOS:
        host = _env(f"DB_HOST_{p}")
        base = _env(f"DB_BASE_{p}")
        if not host or not base:
            continue
        user = _env(f"DB_USER_{p}")
        pwd = _env(f"DB_PASSWORD_{p}")
        port = _env(f"DB_PORT_{p}", "1433")
        conns[p] = build_conn_str(host, base, user, pwd, port, encrypt, trust_cert, timeout)
    return conns


# Conex?es lidas do .env (chave = posto, valor = connection string)
CONNS = build_conns_from_env()


def make_engine(odbc_conn_str: str):
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_conn_str)}",
        pool_pre_ping=True,
    )


def load_sql_for_posto(posto: str) -> str:
    path = os.path.join(SQL_DIR, f"{posto}.sql")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            sql = f.read().strip()
        if sql.endswith(";"):
            sql = sql[:-1]
        return sql

    # Fallback m?nimo; ajuste conforme suas colunas reais
    return (
        "SELECT "
        "p.idprontuario, p.idcliente, p.iddependente, p.matricula, p.idendereco, "
        "p.paciente, p.idade, p.datanascimento, p.peso, p.altura, "
        "p.parterialinicio, p.parterialfim, p.doencascronicas, p.medicaoanterior, "
        "p.queixa, p.conduta, p.desativado, p.datainicioconsulta, p.datafimconsulta, "
        "p.idmedico, p.observacao, p.informacao, p.temperatura, p.bpm, "
        "p.idespecialidade, p.especialidade, p.nomesocial, p.pacienteatendido, "
        "m.Nome AS profissional_atendente "
        "FROM cad_prontuario p LEFT JOIN cad_medico m ON m.idMedico = p.idmedico"
    )


def wrap_with_filters(base_sql: str, use_like: bool) -> text:
    if use_like:
        where_clause = (
            "WHERE LTRIM(RTRIM(q.paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI "
            "LIKE ('%' + LTRIM(RTRIM(:paciente)) + '%') COLLATE SQL_Latin1_General_CP1_CI_AI "
            "AND CAST(q.DataNascimento AS date) = :nasc"
        )
    else:
        where_clause = (
            "WHERE LTRIM(RTRIM(q.paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI = "
            "LTRIM(RTRIM(:paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI "
            "AND CAST(q.DataNascimento AS date) = :nasc"
        )
    return text(f"SELECT * FROM (\n{base_sql}\n) AS q\n{where_clause}")


def query_posto(label: str, odbc_conn_str: str, paciente: str, nasc_date, use_like=False) -> pd.DataFrame:
    print(f"[{label}] Conectando...")
    base_sql = load_sql_for_posto(label)
    sql = wrap_with_filters(base_sql, use_like=use_like)
    params = {"paciente": paciente.strip(), "nasc": nasc_date}
    try:
        engine = make_engine(odbc_conn_str)
        with engine.begin() as con:
            df = pd.read_sql(sql, con=con, params=params)
        if df.empty:
            print(f"[{label}] Nenhum registro")
        else:
            print(f"[{label}] {len(df)} registro(s)")
        return df
    except Exception as e:
        print(f"[{label}] ERRO: {e}")
        return pd.DataFrame()


def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if np.issubdtype(df2[col].dtype, np.datetime64):
            df2[col] = df2[col].astype("datetime64[ns]").dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            df2[col] = df2[col].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    return df2.replace({np.nan: None})


def save_json(payload: dict) -> str:
    ensure_dirs()
    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(JSON_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path

def _load_prompt_file(prompt_path: str) -> tuple[str, str]:
    import hashlib
    # Tenta 4 caminhos em ordem
    candidates = [
        prompt_path,  # expl?cito
        os.path.join(os.path.dirname(__file__), "prompts", "prompt_analise_atds_anteriores.txt"),
        os.path.join(os.getcwd(), "prompts", "prompt_analise_atds_anteriores.txt"),
        os.getenv("PROMPT_PATH", "").strip(),  # override por ambiente
    ]
    errors = []
    for p in [c for c in candidates if c]:
        try:
            ap = os.path.abspath(p)
            if os.path.isfile(ap):
                with open(ap, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
                print(f"[Groq] System prompt carregado: {ap} (sha256={sha256[:12]}...)")
                return content, sha256
            else:
                errors.append(f"n?o existe: {ap}")
        except Exception as e:
            errors.append(f"falha em {p}: {e}")
    print("[Groq] Falha ao localizar prompt. Tentativas: " + " | ".join(errors))
    return "", ""


def analyze_with_groq(summary_text: str,
                      config_path: str = "groq.ini",
                      prompt_path: str = DEFAULT_PROMPT_PATH) -> str:
    """
    Analisa o caso cl?nico via Groq usando configura??es de um arquivo INI e
    injeta o conte?do de ./prompts/prompt_analise_atds_anteriores.txt como SYSTEM.
    """
    # ---------------------------
    # Imports locais
    # ---------------------------
    import ast
    from configparser import ConfigParser

    # ---------------------------
    # API key
    # ---------------------------
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "Groq indispon?vel: defina GROQ_API_KEY no ambiente (.env ou vari?vel de sistema)."

    # ---------------------------
    # Config INI
    # ---------------------------
    cfg = ConfigParser()
    if not cfg.read(config_path, encoding="utf-8"):
        cfg.add_section("groq")
        cfg.set("groq", "model", "openai/gpt-oss-120b")
        cfg.set("groq", "temperature", "0")
        cfg.set("groq", "top_p", "1")
        cfg.set("groq", "max_completion_tokens", "16000")
        cfg.set("groq", "stream", "false")
        cfg.set("groq", "reasoning_effort", "")
        cfg.set("groq", "json_mode", "false")
        cfg.set("groq", "stop", "[]")

    g = cfg["groq"]
    model = g.get("model", "openai/gpt-oss-120b").strip('"').strip("'")
    temperature = float(g.get("temperature", 0))
    top_p = float(g.get("top_p", 1))
    max_tokens = int(g.get("max_completion_tokens", 16000))
    stream = g.get("stream", "false").strip().lower() == "true"
    reasoning_effort = g.get("reasoning_effort", "").strip().strip('"').strip("'")
    json_mode = g.get("json_mode", "false").strip().lower() == "true"

    try:
        stop_val = ast.literal_eval(g.get("stop", "[]"))
        stop = stop_val if isinstance(stop_val, (list, tuple)) else []
    except Exception:
        stop = []

    # ---------------------------
    # Prompt externo
    # ---------------------------
    external_system_prompt, prompt_sha = _load_prompt_file(prompt_path)
    default_system = "Voc? ? um assistente cl?nico objetivo. Responda em portugu?s brasileiro."
    system_prompt = external_system_prompt if external_system_prompt else default_system

    if external_system_prompt:
        print(f"[Groq] System prompt carregado de '{prompt_path}' (sha256={prompt_sha[:12]}...)")
    else:
        print("[Groq] System prompt externo n?o encontrado. Usando prompt padr?o embutido.")

    # ---------------------------
    # Mensagens
    # ---------------------------
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": summary_text},
    ]

    optional_args = {}
    if reasoning_effort:
        optional_args["reasoning"] = {"effort": reasoning_effort}
    if json_mode:
        optional_args["response_format"] = {"type": "json_object"}
    if stop:
        optional_args["stop"] = list(stop)

    # ---------------------------
    # Chamada ? API Groq
    # ---------------------------
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        if stream:
            acc = []
            with client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
                **optional_args,
            ) as stream_resp:
                for chunk in stream_resp:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        acc.append(delta)
            return "".join(acc).strip()

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **optional_args,
        )

        content = resp.choices[0].message.content or ""
        if json_mode:
            try:
                _ = json.loads(content)
            except Exception:
                pass
        return content.strip()

    except Exception as e:
        return f"Falha ao consultar Groq: {e}"


# =========================
# FLUXO PRINCIPAL (CLI)
# =========================
def main():
    ensure_dirs()

    # -------- Inputs --------
    nome = input("1) Nome completo do paciente: ").strip()
    data_nasc_str = input("2) Data de nascimento (dd/mm/yyyy): ").strip()
    queixa = input("3) Queixa atual do paciente: ").strip()
    api_choice = input("API para an?lise? (1=OpenAI, 2=Groq): ").strip()

    # -------- Valida??o da data --------
    try:
        nasc_date = datetime.strptime(data_nasc_str, "%d/%m/%Y").date()
    except ValueError:
        print("Data inv?lida. Use dd/mm/yyyy.")
        return

    # -------- Conex?es --------
    if not CONNS:
        print("Nenhum posto configurado no .env. Preencha DB_HOST_*/DB_BASE_* e rode novamente.")
        return

    # -------- Igualdade exata --------
    frames = []
    for lbl, conn_str in CONNS.items():
        df = query_posto(lbl, conn_str, nome, nasc_date, use_like=False)
        if not df.empty:
            df.insert(0, "posto", lbl)
            frames.append(df)

    frames_nonempty = [df for df in frames if not df.empty]
    df_all = pd.concat(frames_nonempty, ignore_index=True) if frames_nonempty else pd.DataFrame()

    # -------- Fallback LIKE --------
    if df_all.empty:
        print("Nenhum dado com igualdade exata. Tentando busca parcial (LIKE)...")
        frames_like = []
        for lbl, conn_str in CONNS.items():
            df = query_posto(lbl, conn_str, nome, nasc_date, use_like=True)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames_like.append(df)
        frames_like_nonempty = [df for df in frames_like if not df.empty]
        df_all = pd.concat(frames_like_nonempty, ignore_index=True) if frames_like_nonempty else pd.DataFrame()

    # -------- Sem resultados --------
    if df_all.empty:
        print("Nenhum dado encontrado em nenhuma unidade.")
        return

    # -------- Enriquecimento e resumo --------
    df_all["queixa_atual_informada"] = queixa
    print("\nResumo (top 10 linhas):")
    print(df_all.head(10).to_string(index=False))

    # -------- JSON --------
    df_json = sanitize_for_json(df_all)
    payload = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "gerado_em": datetime.now().isoformat(),
            "registros": int(len(df_all)),
        },
        "amostra": df_json.head(200).to_dict(orient="records"),
    }
    json_path = save_json(payload)
    print(f"\nJSON gerado: {json_path}")

    # -------- An?lise opcional --------
    if api_choice == "2":
        cols = [
            c for c in df_all.columns
            if c.lower() in {
                "posto", "paciente", "idade", "datanascimento", "peso", "altura",
                "parterialinicio", "parterialfim", "temperatura", "bpm",
                "queixa", "conduta", "informacao", "observacao",
            }
        ]
        sample = df_all[cols].head(5) if cols else df_all.head(5)
        summary = (
            f"Paciente: {nome}\n"
            f"Nascimento: {nasc_date.isoformat()}\n"
            f"Queixa atual: {queixa}\n"
            f"Dados amostrais:\n{sample.to_string(index=False)}"
        )
        print("\n[A] Enviando para Groq...")
        suggestion = analyze_with_groq(summary, prompt_path=DEFAULT_PROMPT_PATH)
        print("\n[Sugest?o da IA]:")
        print(suggestion)
    elif api_choice == "1":
        print("\nOpenAI selecionado. Integra??o n?o implementada neste script.")

    # -------- Limpeza do JSON --------
    opt = input("\nDeseja apagar o JSON gerado? (s/n): ").strip().lower()
    if opt == "s":
        try:
            os.remove(json_path)
            print("JSON apagado.")
        except Exception as e:
            print(f"Falha ao apagar JSON: {e}")
    else:
        print("JSON mantido.")


if __name__ == "__main__":
    main()
