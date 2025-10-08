# -*- coding: utf-8 -*-
"""
project.py — CLI para buscar prontuários em múltiplos bancos (por “posto”),
gerar JSON SOMENTE com históricos e enviar a análise para a Groq em chunks.
A queixa/atendimento atual é enviada como VARIÁVEIS separadas (sem id).

ATUALIZAÇÕES:
1) Saída no terminal APENAS TEXTO (sem tabela/HTML). Cada bloco/parágrafo separado por 1 linha em branco.
2) NÃO enviar para a Groq atendimentos que contenham "O paciente não compareceu a chamada"
   (variações com acento/maiúsculas/pontuação também são filtradas).
   >>> Mesma regra aplicada ao BLOCO 1 (não listar “não compareceu” nos 10 últimos).
3) Saneamento automático: a cada execução, remove arquivos com >1h em ./json e ./reports.
4) Títulos em CAIXA ALTA e sem repetição. Para o primeiro bloco: imprimir só “ÚLTIMOS 10 ATENDIMENTOS”.
5) A resposta da IA (Groq/OpenAI) é SEMPRE JSON com schema:
   {"bloco2": str, "bloco3": str, "bloco4": str, "rodape": str}

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv groq requests
"""

import os, re, json, uuid, time, argparse, unicodedata
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Tuple, Iterable, Optional, Literal

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from configparser import ConfigParser

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

SQL_DIR      = os.path.join(BASE_DIR, "sql")
JSON_DIR     = os.path.join(BASE_DIR, "json")
PROMPTS_DIR  = os.path.join(BASE_DIR, "prompts")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
DEFAULT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "prompt_analise_atds_anteriores.txt")
DEFAULT_GROQ_CONFIG = os.path.join(BASE_DIR, "groq_modelos", "gpt120b.txt")

POSTOS = ["A", "N", "X", "Y", "B", "R", "P", "C", "D", "G", "I", "J", "M"]

# ---------------- Utils ----------------
def ensure_dirs():
    for d in (SQL_DIR, JSON_DIR, PROMPTS_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)

def purge_old_files(dirpath: str, older_than_hours: int = 1):
    """Remove arquivos mais antigos que N horas (executado a cada run)."""
    if not os.path.isdir(dirpath):
        return
    now = time.time()
    cutoff = now - (older_than_hours * 3600)
    for root, _, files in os.walk(dirpath):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                if os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
            except Exception:
                pass

def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else v

def to_python_scalar(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        try:
            v = float(x)
            return None if np.isnan(v) else v
        except Exception:
            return None
    return x

def to_python_tree(obj):
    if isinstance(obj, dict):
        return {k: to_python_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_tree(v) for v in obj]
    return to_python_scalar(obj)

def clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn")

# ---------------- Conexão ----------------
def build_conn_str(host, base, user, pwd, port, encrypt, trust_cert, timeout) -> str:
    server = f"tcp:{host},{port or '1433'}"
    common = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};DATABASE={base};"
        f"Encrypt={encrypt};TrustServerCertificate={trust_cert};"
        f"Connection Timeout={timeout or '5'}"
    )
    if user:
        return f"{common};UID={user};PWD={pwd}"
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

def make_engine(odbc_conn_str: str):
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_conn_str)}", pool_pre_ping=True)

def load_sql_for_posto(posto: str) -> str:
    path = os.path.join(SQL_DIR, f"{posto}.sql")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            sql = f.read().strip()
        return sql[:-1] if sql.endswith(";") else sql
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
        print(f"[{label}] {('Nenhum registro' if df.empty else str(len(df))+' registro(s)')}")
        return df
    except Exception as e:
        print(f"[{label}] ERRO: {e}")
        return pd.DataFrame()

# ---------------- JSON / Prompt cfg ----------------
def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if np.issubdtype(df2[col].dtype, np.datetime64):
            df2[col] = df2[col].astype("datetime64[ns]").dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            df2[col] = df2[col].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    df2 = df2.replace({np.nan: None})
    for c in df2.columns:
        df2[c] = df2[c].map(to_python_scalar)
    return df2

def save_json(payload: dict) -> str:
    ensure_dirs()
    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(JSON_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python_tree(payload), f, ensure_ascii=False, indent=2, default=str)
    return path

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
    raise last_err if last_err else UnicodeDecodeError("unknown", b"", 0, 1, "decode failed")

def _load_prompt_file(prompt_path: str) -> tuple[str, str]:
    candidates = [
        prompt_path,
        os.path.join(BASE_DIR, "prompts", "prompt_analise_atds_anteriores.txt"),
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
            print(f"[Groq] System prompt: {ap} (encoding={enc}, sha256={sha[:12]}...)")
            return content, sha
        except Exception as e:
            tried.append(f"falha em {ap}: {e}")
    print("[Groq] Falha ao carregar prompt. Tentativas: " + " | ".join(tried))
    return "", ""

def _read_groq_config(path: str) -> tuple[dict, dict]:
    if not path or not os.path.isfile(path):
        path = DEFAULT_GROQ_CONFIG
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
        raise last_err if last_err else RuntimeError("Falha ao ler configuração GROQ.")
    text_norm = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    def _flatten_fields_keep(m):
        inner = m.group(1)
        payload = "[" + inner + "]"
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
            one_line = re.sub(r",\s*\]", "]", one_line)
            parsed = _json.loads(one_line)
        return f'fields_keep = {json.dumps(parsed, ensure_ascii=False)}'

    text_norm = re.sub(r"fields_keep\s*=\s*\[\s*(.*?)\s*\]", _flatten_fields_keep, text_norm,
                       flags=re.DOTALL | re.IGNORECASE)

    parser = ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read_string(text_norm)

    groq = dict(parser.items("groq")) if parser.has_section("groq") else {}
    processing = dict(parser.items("processing")) if parser.has_section("processing") else {}

    groq["model"] = groq.get("model", "").strip().strip('"').strip("'")
    groq["temperature"] = float(groq.get("temperature", "0.2"))
    groq["top_p"] = float(groq.get("top_p", "1"))
    groq["max_completion_tokens"] = int(groq.get("max_completion_tokens", "4000"))
    groq["stream"] = str(groq.get("stream", "false")).lower() == "true"
    groq["reasoning_effort"] = groq.get("reasoning_effort", "").strip().strip('"').strip("'")
    try:
        groq["stop"] = json.loads(groq.get("stop", "[]"))
    except Exception:
        groq["stop"] = []

    processing["chunk_size"] = int(processing.get("chunk_size", "200"))
    processing["sleep_between_calls"] = float(processing.get("sleep_between_calls", "0.8"))
    processing["max_retries"] = int(processing.get("max_retries", "3"))
    processing["retry_backoff"] = float(processing.get("retry_backoff", "2.0"))
    try:
        processing["fields_keep"] = json.loads(processing.get("fields_keep", "[]"))
    except Exception:
        processing["fields_keep"] = []

    return groq, processing

# ---------------- Chunk helpers ----------------
def _is_blank_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([True] * 0)
    return s.isna() | s.astype(str).str.strip().eq("") | s.astype(str).str.lower().eq("none")

def _filter_with_content(df: pd.DataFrame) -> pd.DataFrame:
    q_blank = _is_blank_series(df.get("queixa", pd.Series([None] * len(df))))
    c_blank = _is_blank_series(df.get("conduta", pd.Series([None] * len(df))))
    keep = ~(q_blank & c_blank)
    return df.loc[keep].reset_index(drop=True)

def _iter_chunks(records: List[dict], size: int) -> Iterable[Tuple[int, int, List[dict]]]:
    total = len(records)
    if total == 0:
        return
    n_parts = (total + size - 1) // size
    for i in range(0, total, size):
        yield (i // size) + 1, n_parts, records[i: i + size]

# ---------------- Filtro "não compareceu" ----------------
def _no_show_text(txt: str) -> bool:
    t = strip_accents(txt or "").lower()
    return ("nao compareceu" in t) and ("chamad" in t)

def _no_show_row(row_like) -> bool:
    if isinstance(row_like, dict):
        q = str(row_like.get("queixa", ""))
        o = str(row_like.get("observacao", ""))
        c = str(row_like.get("conduta", ""))
    else:
        q = str(row_like["queixa"]); o = str(row_like["observacao"]); c = str(row_like["conduta"])
    return _no_show_text(" ".join([q, o, c]))

# ---------------- Groq ----------------
def _call_groq(client, model, messages, temperature, top_p, max_tokens, stop=None, reasoning_effort: str = ""):
    kwargs = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    if stop:
        kwargs["stop"] = stop
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return client.chat.completions.create(**kwargs)

def _force_json_output(txt: str) -> dict:
    """Garante dict JSON com chaves bloco2/3/4/rodape (fallback seguro)."""
    if not txt:
        return {"bloco2": "", "bloco3": "", "bloco4": "", "rodape": ""}
    # tentativa direta
    try:
        obj = json.loads(txt)
        for k in ("bloco2", "bloco3", "bloco4", "rodape"):
            obj.setdefault(k, "")
        # mantêm só as chaves de interesse
        return {k: str(obj.get(k, "")) for k in ("bloco2","bloco3","bloco4","rodape")}
    except Exception:
        pass
    # tenta extrair o maior JSON dentro do texto
    m = re.search(r"\{[\s\S]*\}$", txt.strip())
    if m:
        try:
            obj = json.loads(m.group(0))
            for k in ("bloco2", "bloco3", "bloco4", "rodape"):
                obj.setdefault(k, "")
            return {k: str(obj.get(k, "")) for k in ("bloco2","bloco3","bloco4","rodape")}
        except Exception:
            pass
    # último recurso: coloca tudo no rodapé
    return {"bloco2": "", "bloco3": "", "bloco4": "", "rodape": txt.strip()}

def analyze_json_with_groq(full_payload: dict, vars_atual: dict,
                           tabela_texto: str, tabela_bloco1: str,
                           prompt_path: str = DEFAULT_PROMPT_PATH,
                           config_path: str = "") -> str:
    api_key = _env("GROQ_API_KEY", "")
    if not api_key:
        # retorna JSON mínimo
        return json.dumps({
            "bloco2": "", "bloco3": "", "bloco4": "",
            "rodape": "Groq indisponível: defina GROQ_API_KEY no ambiente."
        }, ensure_ascii=False)

    cfg_path = config_path or _env("GROQ_CONFIG", DEFAULT_GROQ_CONFIG)
    groq_cfg, proc_cfg = _read_groq_config(cfg_path)

    from groq import Groq
    client = Groq(api_key=api_key)

    model = (groq_cfg.get("model") or "llama-3.1-70b-versatile").strip()
    temperature = float(groq_cfg.get("temperature", 0.2))
    top_p = float(groq_cfg.get("top_p", 1.0))
    max_tokens = min(int(groq_cfg.get("max_completion_tokens", "4000")), 4000)
    stop = groq_cfg.get("stop") or None
    reasoning_effort = groq_cfg.get("reasoning_effort", "")

    chunk_size = max(int(proc_cfg.get("chunk_size", 200)), 50)
    sleep_between = float(proc_cfg.get("sleep_between_calls", 0.6))
    max_retries = int(proc_cfg.get("max_retries", 3))
    retry_backoff = float(proc_cfg.get("retry_backoff", 2.0))

    external_system_prompt, _ = _load_prompt_file(_env("PROMPT_PATH", prompt_path))
    # Força saída JSON estrita
    system_prompt = (external_system_prompt or "Você é um assistente clínico objetivo. Responda em pt-BR.") + (
        "\n\nSAÍDA OBRIGATÓRIA: responda EXCLUSIVAMENTE em JSON válido, SEM markdown, no schema:\n"
        '{"bloco2": string, "bloco3": string, "bloco4": string, "rodape": string}\n'
        "Não inclua comentários fora do JSON. Não use blocos de código. "
        "Se faltar informação, deixe o campo vazio ou escreva 'NÃO ENCONTRADO NO HISTÓRICO'."
    )

    # ------ HISTÓRICO (com filtros para IA) ------
    records = list(full_payload.get("amostra") or [])
    records = [r for r in records if not _no_show_row(r)]  # remove “não compareceu” para a IA

    df_all = pd.DataFrame.from_records(records) if records else pd.DataFrame()
    df_envio = sanitize_for_json(_filter_with_content(df_all))
    recs_envio = df_envio.to_dict(orient="records")

    print(f"[Groq] Registros no JSON (após filtros): {len(df_envio)}")
    if not recs_envio:
        return json.dumps({
            "bloco2": "", "bloco3": "", "bloco4": "",
            "rodape": "Nenhum registro elegível para análise (após filtros)."
        }, ensure_ascii=False)

    vars_atual = to_python_tree(vars_atual)
    vars_str = json.dumps(vars_atual, ensure_ascii=False)

    header = (
        "VARIÁVEIS DO ATENDIMENTO ATUAL (sem id, não salvo):\n"
        + vars_str +
        "\n\nBLOCO_1_TABELA_FIXA (use EXATAMENTE estas linhas no Bloco 1, sem acrescentar/remover):\n"
        + (tabela_bloco1 or "(sem registros)") +
        "\n\nTABELA_DE_REFERENCIA (apoio para os demais blocos):\n"
        + (tabela_texto or "(sem registros)") +
        "\n\nRegras obrigatórias:\n"
        "1) O Bloco 1 deve repetir a BLOCO_1_TABELA_FIXA exatamente.\n"
        "2) Nos demais blocos, cite (POSTO, IDPRONTUARIO, DATA) e NÃO invente fatos/medicações.\n"
        "3) Compare a queixa atual com os históricos ao listar achados.\n"
        "Responda SEMPRE em JSON no schema definido."
    )

    parts = []
    for k, n, recs in _iter_chunks(recs_envio, chunk_size):
        payload_str = json.dumps(recs, ensure_ascii=False)
        user_msg = (
            f"{header}\n"
            f"Parte {k}/{n} de prontuários históricos (JSON):\n{payload_str}\n\n"
            "Tarefa: Resuma APENAS esta parte em bullets objetivos.\n"
            "- Liste (POSTO, IDPRONTUARIO, DATA) + queixa/conduta relevantes.\n"
            "- Relacione explicitamente com a queixa atual.\n"
            "- NÃO invente fatos/medicações. Se não constar, diga 'não consta'.\n"
            "NÃO RESPONDA AGORA; aguarde a consolidação final."
        )
        attempt = 0
        while True:
            try:
                resp = _call_groq(
                    client, model,
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_msg}],
                    temperature, top_p, max_tokens, stop, reasoning_effort
                )
                parts.append(f"### Parte {k}/{n}\n{resp.choices[0].message.content.strip()}")
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    return json.dumps({
                        "bloco2": "", "bloco3": "", "bloco4": "",
                        "rodape": f"[Groq] Erro no chunk {k}/{n}: {e}"
                    }, ensure_ascii=False)
                wait = (retry_backoff ** (attempt - 1))
                print(f"[Groq] Falha no chunk {k}/{n} ({e}). Retry {attempt}/{max_retries} em {wait:.1f}s.")
                time.sleep(wait)
        if sleep_between > 0:
            time.sleep(sleep_between)

    final_user = (
        f"Considere as VARIÁVEIS DO ATENDIMENTO ATUAL:\n{vars_str}\n\n"
        "Consolide os resumos parciais em um relatório ÚNICO no formato:\n"
        '- "bloco2": texto da seção "Queixa atual nos atendimentos anteriores"\n'
        '- "bloco3": texto da seção "Diagnóstico diferencial"\n'
        '- "bloco4": texto da seção "Sugestão de OBSERVAÇÃO e CONDUTA"\n'
        '- "rodape": itens sem evidência textual explícita\n\n'
        "Responda SOMENTE com JSON válido no schema {bloco2,bloco3,bloco4,rodape}. Sem markdown, sem comentários.\n\n"
        + "\n\n---\n\n".join(parts)
    )
    try:
        final = _call_groq(
            client, model,
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": final_user}],
            temperature, top_p, max_tokens, stop, reasoning_effort
        )
        content = final.choices[0].message.content.strip()
        # força JSON
        blocks = _force_json_output(content)
        return json.dumps(blocks, ensure_ascii=False)
    except Exception as e:
        print(f"[Groq] Falha na consolidação ({e}). Retornando fallback JSON.")
        return json.dumps({
            "bloco2": "", "bloco3": "", "bloco4": "",
            "rodape": f"# Consolidação (fallback)\n\n" + "\n\n".join(parts)
        }, ensure_ascii=False)

# ---------------- Parsing da resposta da IA (JSON -> dict) ----------------
def ai_text_to_blocks(ai_text: str) -> dict:
    """Converte texto da IA em dict com chaves bloco2/3/4/rodape (sempre)."""
    try:
        obj = json.loads(ai_text)
        return {
            "bloco2": str(obj.get("bloco2", "")),
            "bloco3": str(obj.get("bloco3", "")),
            "bloco4": str(obj.get("bloco4", "")),
            "rodape": str(obj.get("rodape", "")),
        }
    except Exception:
        # último recurso
        return {"bloco2": "", "bloco3": "", "bloco4": "", "rodape": ai_text.strip()}

def paragraphs_text(md: str) -> str:
    """Texto simples (mantém bullets), sem linhas vazias múltiplas."""
    if not md or not md.strip():
        return "NÃO ENCONTRADO NO HISTÓRICO"
    lines = [ln.rstrip() for ln in md.split("\n") if ln.strip()]
    return "\n".join(lines)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Busca de prontuários multi-postos + análise Groq.")
    p.add_argument("-n", "--nome", help="Nome completo do paciente")
    p.add_argument("-d", "--nascimento", help="Data de nascimento dd/mm/yyyy")
    p.add_argument("-q", "--queixa", help="Queixa atual")
    p.add_argument("-a", "--api", choices=["openai", "groq", "1", "2"], help="Provedor de análise")
    p.add_argument("--like", action="store_true", help="Se não achar por igualdade, tenta LIKE automaticamente")
    p.add_argument("--no-delete-json", action="store_true", help="Não pergunta e mantém o JSON")
    p.add_argument("--prompt-path", default=_env("PROMPT_PATH", DEFAULT_PROMPT_PATH), help="Caminho do prompt")
    p.add_argument("--groq-config", default=_env("GROQ_CONFIG", DEFAULT_GROQ_CONFIG), help="Caminho do INI da Groq")
    return p.parse_args()

def build_last10(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    # aplica filtro “não compareceu” TAMBÉM no Bloco 1
    mask_no_show = df.apply(lambda r: _no_show_row(r), axis=1)
    df = df.loc[~mask_no_show].copy()

    df["_dt_ini"] = pd.to_datetime(df["datainicioconsulta"], errors="coerce")
    df = (
        df.loc[df["_dt_ini"].notna(),
               ["idprontuario","posto","_dt_ini","queixa","observacao","conduta"]]
        .sort_values("_dt_ini", ascending=False).head(10).copy()
    )
    df["data"] = df["_dt_ini"].dt.strftime("%d/%m/%Y")

    def _mk_resumo(r):
        partes = [clean_text(r["queixa"]), clean_text(r["observacao"]), clean_text(r["conduta"])]
        partes = [p for p in partes if p]
        if not partes: return ""
        s = ". ".join(partes).strip(" .")
        return s + "."
    df["resumo"] = df.apply(_mk_resumo, axis=1)
    return df

def main():
    ensure_dirs()
    # saneamento a cada execução (remove arquivos >1h)
    purge_old_files(JSON_DIR, 1)
    purge_old_files(REPORTS_DIR, 1)

    args = parse_args()

    nome = args.nome or input("1) Nome completo do paciente: ").strip()
    data_nasc_str = args.nascimento or input("2) Data de nascimento (dd/mm/yyyy): ").strip()
    queixa = args.queixa or input("3) Queixa atual do paciente: ").strip()
    api_choice = args.api or input("API para análise? (1=OpenAI, 2=Groq): ").strip()
    api_choice = {"1": "openai", "2": "groq"}.get(api_choice, api_choice)

    try:
        nasc_date = datetime.strptime(data_nasc_str, "%d/%m/%Y").date()
    except ValueError:
        print("Data inválida. Use dd/mm/yyyy.")
        return

    conns = build_conns_from_env()
    if not conns:
        print("Nenhum posto configurado no .env.")
        return

    frames = []
    for lbl, conn_str in conns.items():
        df = query_posto(lbl, conn_str, nome, nasc_date, use_like=False)
        if not df.empty:
            df.insert(0, "posto", lbl)
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df_all.empty and (args.like or True):
        print("Nenhum dado com igualdade exata. Tentando busca parcial (LIKE)...")
        frames_like = []
        for lbl, conn_str in conns.items():
            df = query_posto(lbl, conn_str, nome, nasc_date, use_like=True)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames_like.append(df)
        df_all = pd.concat(frames_like, ignore_index=True) if frames_like else pd.DataFrame()

    if df_all.empty:
        print("Nenhum dado encontrado em nenhuma unidade.")
        return

    # Normalização mínima
    df_all.columns = [str(c).strip().lower() for c in df_all.columns]
    for col in ["queixa", "observacao", "conduta", "especialidade", "profissional_atendente", "datainicioconsulta"]:
        if col not in df_all.columns:
            df_all[col] = None
    for col in ["queixa", "observacao", "conduta", "especialidade", "profissional_atendente"]:
        df_all[col] = df_all[col].map(clean_text)

    # ----- BLOCO 1 (últimos 10) -----
    df_b1 = build_last10(df_all)
    linhas_b1 = ["idprontuario | posto | data | resumo"]
    for _, r in df_b1.iterrows():
        idp = str(r["idprontuario"]) if r["idprontuario"] is not None else ""
        linhas_b1.append(f'{idp} | {r["posto"]} | {r["data"]} | {r["resumo"]}')
    tabela_bloco1_txt = "\n".join(linhas_b1)

    # Salva cópia do bloco 1 (TXT) — será limpo em <=1h
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    arq_b1 = os.path.join(REPORTS_DIR, f"bloco1_{ts}.txt")
    with open(arq_b1, "w", encoding="utf-8") as f:
        f.write(tabela_bloco1_txt)

    # Tabela de referência (ISO + especialidade) para apoio da IA
    def _to_iso(x):
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return str(x) if pd.notna(x) else "—"
    cols_tab = ["idprontuario","posto","datainicioconsulta","especialidade","queixa","observacao","conduta"]
    for c in cols_tab:
        if c not in df_all.columns: df_all[c] = ""
    df_tab = df_all[cols_tab].copy()
    df_tab["datainicioconsulta"] = df_tab["datainicioconsulta"].apply(_to_iso)
    df_tab = df_tab.sort_values("datainicioconsulta", ascending=False).head(10).reset_index(drop=True)


    linhas_ref = ["idprontuario | posto | datainicioconsulta | especialidade | resumo_curto"]
    for _, r in df_tab.iterrows():
        resumo = " / ".join([clean_text(r["queixa"]), clean_text(r["observacao"]), clean_text(r["conduta"])])
        resumo = re.sub(r"\s+/ \s*", " / ", resumo).strip(" /")
        linhas_ref.append(f'{r["idprontuario"]} | {r["posto"]} | {r["datainicioconsulta"]} | {r["especialidade"] or "—"} | {resumo}')
    tabela_texto_ref = "\n".join(linhas_ref)

    # VARS ATUAIS
    vars_atual = {
        "nome_paciente": nome,
        "data_nascimento": nasc_date.isoformat(),
        "queixa_atual": queixa,
        "data_hora_atendimento": datetime.now().isoformat(),
    }
    vars_atual = to_python_tree(vars_atual)

    # JSON (somente históricos)
    payload = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "gerado_em": datetime.now().isoformat(),
            "registros": int(len(df_all)),
        },
        "amostra": sanitize_for_json(df_all).to_dict(orient="records"),
    }
    json_path = save_json(payload)
    print(f"JSON: {os.path.basename(json_path)} (será limpo em ~1h)")

    # Análise IA
    if api_choice == "groq":
        cfg_path = args.groq_config
        print(f"[A] Enviando para Groq: {cfg_path}")
        try:
            sugestao_ia = analyze_json_with_groq(
                payload,
                vars_atual=vars_atual,
                tabela_texto=tabela_texto_ref,
                tabela_bloco1=tabela_bloco1_txt,
                prompt_path=args.prompt_path,
                config_path=cfg_path,
            )
        except Exception as e:
            sugestao_ia = json.dumps({"bloco2":"","bloco3":"","bloco4":"","rodape":f"[ERRO] Falha na análise Groq: {e}"},
                                     ensure_ascii=False)
    elif api_choice == "openai":
        # ainda não implementado: devolve JSON válido
        sugestao_ia = json.dumps({"bloco2":"","bloco3":"","bloco4":"",
                                  "rodape":"[INFO] Integração OpenAI não implementada neste CLI."},
                                 ensure_ascii=False)
    else:
        sugestao_ia = json.dumps({"bloco2":"","bloco3":"","bloco4":"",
                                  "rodape":"[ERRO] Provedor inválido. Use 'groq' ou 'openai'."},
                                 ensure_ascii=False)

    # --------- SAÍDA NO TERMINAL (APENAS TEXTO) ---------
    blocos = ai_text_to_blocks(sugestao_ia)

    out_lines = []
    out_lines.append("ÚLTIMOS 10 ATENDIMENTOS")
    out_lines.append(tabela_bloco1_txt)
    out_lines.append("")
    out_lines.append("QUEIXA ATUAL NOS ATENDIMENTOS ANTERIORES")
    out_lines.append(paragraphs_text(blocos.get("bloco2","")))
    out_lines.append("")
    out_lines.append("DIAGNÓSTICO DIFERENCIAL")
    out_lines.append(paragraphs_text(blocos.get("bloco3","")))
    out_lines.append("")
    out_lines.append("SUGESTÃO DE OBSERVAÇÃO E CONDUTA")
    out_lines.append(paragraphs_text(blocos.get("bloco4","")))
    out_lines.append("")
    out_lines.append("RODAPÉ")
    out_lines.append(paragraphs_text(blocos.get("rodape","")))
    out_lines.append("")

    print("\n".join(out_lines))

    # -------- Limpeza opcional imediata do JSON --------
    if args.no_delete_json:
        print("JSON mantido por opção de linha de comando (--no-delete-json).")
        return
    opt = input("Deseja apagar o JSON gerado agora? (s/n): ").strip().lower()
    if opt == "s":
        try:
            os.remove(json_path)
            print("JSON apagado.")
        except Exception as e:
            print(f"Falha ao apagar JSON: {e}")
    else:
        print("JSON mantido (será limpo automaticamente em ~1h).")

# ---------------- API helper (texto simples) ----------------
def executar_pipeline_api(
    nome: str, data_nascimento_mmddyyyy: str, queixa: str,
    provedor: Literal["groq", "openai"] = "groq", modelo: Optional[str] = None, like: bool = False
) -> dict:
    ensure_dirs()
    purge_old_files(JSON_DIR, 1)
    purge_old_files(REPORTS_DIR, 1)

    try:
        nasc_date = datetime.strptime(data_nascimento_mmddyyyy, "%m/%d/%Y").date()
    except ValueError as e:
        return {"ok": False, "erro": f"Data inválida: {e}. Use mm/dd/yyyy."}

    conns = build_conns_from_env()
    if not conns:
        return {"ok": False, "erro": "Nenhum posto configurado (.env)."}

    frames = []
    for lbl, conn_str in conns.items():
        df = query_posto(lbl, conn_str, nome, nasc_date, use_like=False)
        if not df.empty:
            df.insert(0, "posto", lbl)
            frames.append(df)
    if not frames and like:
        for lbl, conn_str in conns.items():
            df = query_posto(lbl, conn_str, nome, nasc_date, use_like=True)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames.append(df)
    if not frames:
        return {"ok": True, "registros": 0, "texto": "ÚLTIMOS 10 ATENDIMENTOS\nNenhum dado encontrado.\n"}

    # Normaliza
    df_all = pd.concat(frames, ignore_index=True)
    df_all.columns = [str(c).strip().lower() for c in df_all.columns]
    for col in ["queixa","observacao","conduta","especialidade","profissional_atendente","datainicioconsulta"]:
        if col not in df_all.columns: df_all[col] = None
    for col in ["queixa","observacao","conduta","especialidade","profissional_atendente"]:
        df_all[col] = df_all[col].map(clean_text)

    # Bloco 1 (com filtro “não compareceu”)
    df_b1 = build_last10(df_all)
    linhas_b1 = ["idprontuario | posto | data | resumo"]
    for _, r in df_b1.iterrows():
        idp = str(r["idprontuario"]) if r["idprontuario"] is not None else ""
        linhas_b1.append(f'{idp} | {r["posto"]} | {r["data"]} | {r["resumo"]}')
    tabela_bloco1_txt = "\n".join(linhas_b1)

    payload = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "gerado_em": datetime.now().isoformat(),
            "registros": int(len(df_all)),
        },
        "amostra": sanitize_for_json(df_all).to_dict(orient="records"),
    }
    vars_atual = {"nome_paciente": nome, "data_nascimento": nasc_date.isoformat(),
                  "queixa_atual": queixa, "data_hora_atendimento": datetime.now().isoformat()}

    if provedor == "groq":
        sugestao_ia = analyze_json_with_groq(
            payload,
            vars_atual=vars_atual,
            tabela_texto="",  # simples
            tabela_bloco1=tabela_bloco1_txt,
            prompt_path=_env("PROMPT_PATH", DEFAULT_PROMPT_PATH),
            config_path=_env("GROQ_CONFIG", DEFAULT_GROQ_CONFIG),
        )
    else:
        # OpenAI ainda não implementado -> garante JSON
        sugestao_ia = json.dumps({"bloco2":"","bloco3":"","bloco4":"",
                                  "rodape":"[INFO] Integração OpenAI não implementada."},
                                 ensure_ascii=False)

    blocos = ai_text_to_blocks(sugestao_ia)
    parts = []
    parts.append("ÚLTIMOS 10 ATENDIMENTOS")
    parts.append(tabela_bloco1_txt)
    parts.append("")
    parts.append("QUEIXA ATUAL NOS ATENDIMENTOS ANTERIORES")
    parts.append(paragraphs_text(blocos.get("bloco2","")))
    parts.append("")
    parts.append("DIAGNÓSTICO DIFERENCIAL")
    parts.append(paragraphs_text(blocos.get("bloco3","")))
    parts.append("")
    parts.append("SUGESTÃO DE OBSERVAÇÃO E CONDUTA")
    parts.append(paragraphs_text(blocos.get("bloco4","")))
    parts.append("")
    parts.append("RODAPÉ")
    parts.append(paragraphs_text(blocos.get("rodape","")))
    parts.append("")
    return {"ok": True, "registros": int(len(df_all)), "texto": "\n".join(parts)}

# ---------------- Entry ----------------
if __name__ == "__main__":
    main()
