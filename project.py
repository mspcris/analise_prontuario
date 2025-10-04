# -*- coding: utf-8 -*-
"""
project.py
Objetivo: Buscar prontu?rios em m?ltiplos bancos (por "posto"), usando SQLs externos
salvos em ./sql/<POSTO>.sql, aplicar filtros case/acento-insens?veis, consolidar resultados,
persistir um JSON em ./json e enviar TODO o conte?do relevante para a Groq em CHUNKS,
excluindo somente registros onde queixa e conduta s?o ambos nulos/vazios.
Ao final, perguntar se deve apagar o JSON.

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv groq
  ODBC Driver 17/18 for SQL Server instalado no Windows

Configura??o:
  Vari?veis de ambiente no .env (na raiz do projeto), ex.:
    DB_HOST_A, DB_PORT_A, DB_BASE_A, DB_USER_A, DB_PASSWORD_A
    DB_ENCRYPT=yes|no, DB_TRUST_CERT=yes|no, DB_TIMEOUT=5
    GROQ_API_KEY=<sua chave>
    PROMPT_PATH=prompts/prompt_analise_atds_anteriores.txt  (opcional)
    GROQ_CONFIG=groq_modelos/gpt120b.txt                    (opcional)

Observa??o:
  O arquivo groq_modelos/gpt120b.txt ? um INI com se??es [groq] e [processing].
"""

# =========================
# IMPORTS E CAMINHOS
# =========================
import os
import json
import uuid
import time
import re
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from configparser import ConfigParser

# Diret?rios base do projeto, SQLs e JSONs
BASE_DIR = os.path.dirname(__file__)
SQL_DIR = os.path.join(BASE_DIR, "sql")
JSON_DIR = os.path.join(BASE_DIR, "json")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
DEFAULT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "prompt_analise_atds_anteriores.txt")
DEFAULT_GROQ_CONFIG = os.path.join(BASE_DIR, "groq_modelos", "gpt120b.txt")

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


# Postos suportados. Edite conforme seu ambiente real.
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
    """
    Converte datetimes p/ ISO e NaN p/ None.
    """
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


# ---------- Prompt loader ----------
def _read_text_auto(file_path: str) -> tuple[str, str]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    enc_candidates = []
    env_enc = os.getenv("PROMPT_ENCODING", "").strip()
    if env_enc:
        enc_candidates.append(env_enc)
    enc_candidates += ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in enc_candidates:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read(), enc
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else UnicodeDecodeError("unknown", b"", 0, 1, "decode failed")


def _load_prompt_file(prompt_path: str) -> tuple[str, str]:
    candidates = [
        prompt_path,
        os.path.join(os.path.dirname(__file__), "prompts", "prompt_analise_atds_anteriores.txt"),
        os.path.join(os.getcwd(), "prompts", "prompt_analise_atds_anteriores.txt"),
        os.getenv("PROMPT_PATH", "").strip(),
    ]
    tried = []
    for p in [c for c in candidates if c]:
        ap = os.path.abspath(p)
        try:
            content, enc = _read_text_auto(ap)
            content = content.strip()
            from hashlib import sha256
            sha = sha256(content.encode("utf-8")).hexdigest()
            print(f"[Groq] System prompt carregado: {ap} (encoding={enc}, sha256={sha[:12]}...)")
            return content, sha
        except Exception as e:
            tried.append(f"falha em {ap}: {e}")
    print("[Groq] Falha ao localizar/abrir prompt. Tentativas: " + " | ".join(tried))
    return "", ""


# ---------- Config reader robusto para groq_modelos/gpt120b.txt ----------
def _read_groq_config(path: str) -> tuple[dict, dict]:
    """
    L? INI com toler?ncia a:
      - coment?rios inline (# ou ;)
      - lista multilinha em processing.fields_keep = [ ... ]
    Retorna (cfg_groq, cfg_processing) normalizados.
    """
    if not path or not os.path.isfile(path):
        path = DEFAULT_GROQ_CONFIG

    # 1) L? texto cru
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    raw = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except Exception as e:
            last_err = e
    if raw is None:
        raise last_err if last_err else RuntimeError("Falha ao ler configura??o GROQ.")

    # 2) Normaliza e remove BOM
    text_norm = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    # 3) Achata fields_keep multiline ? JSON em uma linha
    fields_json_list = []
    def _flatten_fields_keep(match: re.Match) -> str:
        inner = match.group(1)
        payload = "[" + inner + "]"
        # remove coment?rios inline
        lines = []
        for line in payload.split("\n"):
            line = re.split(r"\s[#;].*$", line)[0]
            if line.strip():
                lines.append(line)
        one_line = " ".join(lines)
        import json as _json
        try:
            parsed = _json.loads(one_line)
        except Exception:
            # remove trailing comma antes de ]
            one_line = re.sub(r",\s*\]", "]", one_line)
            parsed = _json.loads(one_line)
        fields_json_list.clear()
        fields_json_list.extend(parsed)
        return f'fields_keep = {json.dumps(parsed, ensure_ascii=False)}'

    text_norm = re.sub(
        r"fields_keep\s*=\s*\[\s*(.*?)\s*\]",
        _flatten_fields_keep,
        text_norm,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 4) Parse com coment?rios inline
    parser = ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read_string(text_norm)

    groq = dict(parser.items("groq")) if parser.has_section("groq") else {}
    processing = dict(parser.items("processing")) if parser.has_section("processing") else {}

    # Normaliza??es
    groq["model"] = groq.get("model", "").strip().strip('"').strip("'")
    groq["temperature"] = float(groq.get("temperature", "0"))
    groq["top_p"] = float(groq.get("top_p", "1"))
    groq["max_completion_tokens"] = int(groq.get("max_completion_tokens", "16000"))
    groq["stream"] = str(groq.get("stream", "false")).lower() == "true"
    groq["reasoning_effort"] = groq.get("reasoning_effort", "").strip().strip('"').strip("'")
    groq["json_mode"] = str(groq.get("json_mode", "false")).lower() == "true"
    try:
        groq["stop"] = json.loads(groq.get("stop", "[]"))
    except Exception:
        groq["stop"] = []

    processing["chunk_size"] = int(processing.get("chunk_size", "200"))
    processing["sleep_between_calls"] = float(processing.get("sleep_between_calls", "0.8"))
    processing["max_retries"] = int(processing.get("max_retries", "3"))
    processing["retry_backoff"] = float(processing.get("retry_backoff", "2.0"))
    # fields_keep vira lista python
    if fields_json_list:
        processing["fields_keep"] = fields_json_list
    else:
        try:
            processing["fields_keep"] = json.loads(processing.get("fields_keep", "[]"))
        except Exception:
            processing["fields_keep"] = []

    return groq, processing


# ---------- Filtro para envio ----------
def _is_blank_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([True] * 0)
    return s.isna() | s.astype(str).str.strip().eq("") | s.astype(str).str.lower().eq("none")


def _filter_with_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mant?m linhas em que pelo menos um entre 'queixa' e 'conduta' possui conte?do.
    Descarta quando ambos vazios/nulos.
    """
    q_blank = _is_blank_series(df.get("queixa", pd.Series([None] * len(df))))
    c_blank = _is_blank_series(df.get("conduta", pd.Series([None] * len(df))))
    keep = ~(q_blank & c_blank)
    return df.loc[keep].reset_index(drop=True)


# ---------- Chunking + chamada Groq ----------
def _iter_chunks(records: List[dict], size: int) -> Iterable[Tuple[int, int, List[dict]]]:
    total = len(records)
    if total == 0:
        return
    n_parts = (total + size - 1) // size
    for i in range(0, total, size):
        yield (i // size) + 1, n_parts, records[i: i + size]


def _call_groq(client, model, messages, temperature, top_p, max_tokens, stop=None, reasoning_effort=""):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop:
        kwargs["stop"] = stop
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return client.chat.completions.create(**kwargs)


def analyze_json_with_groq(full_payload: dict,
                           prompt_path: str = DEFAULT_PROMPT_PATH,
                           config_path: str = "") -> str:
    """
    Envia TODO o conte?do do JSON para a Groq, em chunks, excluindo
    apenas registros com queixa e conduta ambos nulos/vazios.
    Modelo e par?metros lidos do arquivo em groq_modelos/.
    """
    api_key = _env("GROQ_API_KEY", "")
    if not api_key:
        return "Groq indispon?vel: defina GROQ_API_KEY no ambiente."

    cfg_path = config_path or _env("GROQ_CONFIG", DEFAULT_GROQ_CONFIG)
    groq_cfg, proc_cfg = _read_groq_config(cfg_path)

    from groq import Groq
    client = Groq(api_key=api_key)

    model = groq_cfg.get("model") or "openai/gpt-oss-120b"
    temperature = groq_cfg["temperature"]
    top_p = groq_cfg["top_p"]
    max_tokens = groq_cfg["max_completion_tokens"]
    stop = groq_cfg.get("stop") or None
    reasoning_effort = groq_cfg.get("reasoning_effort", "")

    chunk_size = proc_cfg["chunk_size"]
    sleep_between = proc_cfg["sleep_between_calls"]
    max_retries = proc_cfg["max_retries"]
    retry_backoff = proc_cfg["retry_backoff"]

    # Prompt
    external_system_prompt, _ = _load_prompt_file(_env("PROMPT_PATH", prompt_path))
    system_prompt = external_system_prompt or "Voc? ? um assistente cl?nico objetivo. Responda em pt-BR."

    # Registros
    records: List[dict] = list(full_payload.get("amostra") or [])
    df_all = pd.DataFrame.from_records(records) if records else pd.DataFrame()
    df_envio = _filter_with_content(df_all)

    print(f"[Groq] Registros no JSON: {len(df_all)} | Enviados (p?s-filtro queixa/conduta): {len(df_envio)}")
    if df_envio.empty:
        return "Nenhum registro eleg?vel para an?lise (queixa e conduta vazios em todos)."

    parts = []
    # Chunks
    for k, n, recs in _iter_chunks(df_envio.to_dict(orient="records"), chunk_size):
        payload_str = json.dumps(recs, ensure_ascii=False)
        user_msg = (
            f"Parte {k}/{n} de prontu?rios (JSON):\n{payload_str}\n\n"
            "Tarefa: Resuma APENAS esta parte em bullets objetivos. "
            "Liste idprontuario, datas (datainicioconsulta/datafimconsulta), posto, "
            "queixa/conduta relevantes e rela??o com 'queixa_atual_informada' quando existir."
        )
        attempt = 0
        while True:
            try:
                resp = _call_groq(
                    client, model,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature, top_p, max_tokens, stop, reasoning_effort
                )
                parts.append(f"### Parte {k}/{n}\n{resp.choices[0].message.content.strip()}")
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = (retry_backoff ** (attempt - 1))
                print(f"[Groq] Falha no chunk {k}/{n} ({e}). Retry {attempt}/{max_retries} em {wait:.1f}s.")
                time.sleep(wait)
        if sleep_between > 0:
            time.sleep(sleep_between)

    # Consolida??o final
    final_user = (
        "Consolide os resumos parciais abaixo em um relat?rio ?nico no formato:\n"
        "1) Resumo executivo\n2) Linha do tempo por posto/data\n3) Achados relevantes\n"
        "4) Lacunas de dados\n5) Pr?ximos passos\n\n"
        + "\n\n---\n\n".join(parts)
    )
    attempt = 0
    while True:
        try:
            final = _call_groq(
                client, model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user},
                ],
                temperature, top_p, max_tokens, stop, reasoning_effort
            )
            return final.choices[0].message.content.strip()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = (retry_backoff ** (attempt - 1))
            print(f"[Groq] Falha na consolida??o ({e}). Retry {attempt}/{max_retries} em {wait:.1f}s.")
            time.sleep(wait)


# =========================
# FLUXO PRINCIPAL (CLI)
# =========================
def main():
    """Orquestra input, consultas, consolida??o, JSON e an?lise."""
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

    # -------- Verifica conex?es --------
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

    # -------- JSON: salvar TUDO --------
    df_json = sanitize_for_json(df_all)
    payload = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "gerado_em": datetime.now().isoformat(),
            "registros": int(len(df_all)),
        },
        # TUDO para auditoria. Sem .head().
        "amostra": df_json.to_dict(orient="records"),
    }
    json_path = save_json(payload)
    print(f"\nJSON gerado: {json_path}")

    # -------- An?lise Groq: envia TUDO em chunks, filtrando s? linhas sem conte?do --------
    if api_choice == "2":
        cfg_path = _env("GROQ_CONFIG", DEFAULT_GROQ_CONFIG)
        print(f"\n[A] Enviando JSON para Groq com config: {cfg_path}")
        try:
            suggestion = analyze_json_with_groq(payload, prompt_path=_env("PROMPT_PATH", DEFAULT_PROMPT_PATH), config_path=cfg_path)
            print("\n[Sugest?o da IA]:")
            print(suggestion)
        except Exception as e:
            print(f"\n[ERRO] Falha na an?lise Groq: {e}")
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


# Entry point
if __name__ == "__main__":
    main()


# --- API ---
from typing import Optional, Literal

def executar_pipeline_api(
    nome: str,
    data_nascimento_mmddyyyy: str,
    queixa: str,
    provedor: Literal["groq", "openai"] = "groq",
    modelo: Optional[str] = None,
    like: bool = False,
) -> dict:
    """
    Orquestra o fluxo existente e retorna um dict para a API.
    - data_nascimento_mmddyyyy: "mm/dd/yyyy" (converte p/ date)
    - provedor: "groq" usa analyze_json_with_groq; "openai" usa REST compat?vel.
    """
    ensure_dirs()
    # 1) Data: mm/dd/yyyy -> date
    try:
        nasc_date = datetime.strptime(data_nascimento_mmddyyyy, "%m/%d/%Y").date()
    except ValueError as e:
        return {"ok": False, "erro": f"Data inv?lida: {e}. Use mm/dd/yyyy."}

    if not CONNS:
        return {"ok": False, "erro": "Nenhum posto configurado (.env DB_HOST_*/DB_BASE_*)."}

    # 2) Busca
    frames = []
    for lbl, conn_str in CONNS.items():
        df = query_posto(lbl, conn_str, nome, nasc_date, use_like=False)
        if not df.empty:
            df.insert(0, "posto", lbl)
            frames.append(df)

    if not frames and like:
        for lbl, conn_str in CONNS.items():
            df = query_posto(lbl, conn_str, nome, nasc_date, use_like=True)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames.append(df)

    if not frames:
        return {"ok": True, "registros": 0, "analise": "Nenhum dado encontrado em nenhuma unidade."}

    df_all = pd.concat(frames, ignore_index=True)
    df_all["queixa_atual_informada"] = queixa

    # 3) JSON integral
    df_json = sanitize_for_json(df_all)
    payload = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "gerado_em": datetime.now().isoformat(),
            "registros": int(len(df_all)),
        },
        "amostra": df_json.to_dict(orient="records"),
    }
    json_path = save_json(payload)

    # 4) An?lise
    analise_texto = ""
    if provedor == "groq":
        analise_texto = analyze_json_with_groq(
            payload,
            prompt_path=_env("PROMPT_PATH", DEFAULT_PROMPT_PATH),
            config_path=_env("GROQ_CONFIG", DEFAULT_GROQ_CONFIG),
        )
        prov = "groq"
        modelo_usado = _read_groq_config(_env("GROQ_CONFIG", DEFAULT_GROQ_CONFIG))[0].get("model") or "openai/gpt-oss-120b"
    elif provedor == "openai":
        # Implementa??o leve via REST
        import requests as _rq
        OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
        if not OPENAI_API_KEY:
            return {"ok": False, "erro": "OPENAI_API_KEY n?o configurada"}
        system_prompt = (_load_prompt_file(_env("PROMPT_PATH", DEFAULT_PROMPT_PATH))[0]
                         or "Voc? ? um assistente cl?nico objetivo. Responda em pt-BR.")
        # Reaproveita o mesmo filtro+chunk
        records = pd.DataFrame.from_records(payload["amostra"])
        records = _filter_with_content(records).to_dict(orient="records")
        chunks = list(_iter_chunks(records, 200))
        partes = []
        for k, n, recs in chunks:
            user_msg = (
                f"Parte {k}/{n} de prontu?rios (JSON):\n{json.dumps(recs, ensure_ascii=False)}\n\n"
                "Tarefa: Resuma APENAS esta parte em bullets objetivos. "
                "Liste idprontuario, datas, posto, queixa/conduta e rela??o com 'queixa_atual_informada'."
            )
            resp = _rq.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": modelo or "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": user_msg}],
                    "temperature": 0.2,
                },
                timeout=120,
            )
            resp.raise_for_status()
            partes.append(f"### Parte {k}/{n}\n{resp.json()['choices'][0]['message']['content'].strip()}")
        final_msg = (
            "Consolide os resumos parciais abaixo em um relat?rio ?nico no formato:"
            " 1) Resumo executivo 2) Linha do tempo por posto/data 3) Achados 4) Lacunas 5) Pr?ximos passos\n\n"
            + "\n\n---\n\n".join(partes)
        )
        resp2 = _rq.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": modelo or "gpt-4o-mini",
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": final_msg}],
                "temperature": 0.2,
            },
            timeout=120,
        )
        resp2.raise_for_status()
        analise_texto = resp2.json()["choices"][0]["message"]["content"].strip()
        prov = "openai"
        modelo_usado = modelo or "gpt-4o-mini"
    else:
        return {"ok": False, "erro": "Provedor inv?lido. Use 'groq' ou 'openai'."}

    return {
        "ok": True,
        "provedor": prov,
        "modelo": modelo_usado,
        "registros": int(len(df_all)),
        "json_path": json_path,
        "analise": analise_texto,
    }
