# -*- coding: utf-8 -*-
"""
project.py — Busca prontuários multi-postos, coleta resultados de exames (DB + CSV fallback) e gera análise via Groq.
- PROMPT: ./prompts/prompt.txt
- MODELOS (.ini): ./groq_modelos/*.ini (ordem pelo prefixo numérico 1-*, 2-*, 3-*, 4-*)
- Consultas por posto (A,N,X,...) usando ./sql/select_prontuarios.sql (único para todos)
- EXAMES via ./sql/select_resultado_exames.sql (único para todos)
- Pergunta interativa: "Buscar desde (dd/mm/yyyy)" — filtra de dt_ini em diante.
- Envia histórico em CHUNKS (map→reduce), inclui EXAMES (compactados). Bloco 1 é narrativa pronta.
- Saída: APENAS JSON (inclui "EXAMES_RESULTADOS", "exames_csv" e "BLOCO1_RESUMO").

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv requests python-dateutil
"""

import os, re, json, uuid, time, argparse, unicodedata, glob
from datetime import datetime, date
from urllib.parse import quote_plus
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from configparser import ConfigParser
import requests

# ---------------- Paths / ENV ----------------
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

SQL_DIR         = os.path.join(BASE_DIR, "sql")
JSON_DIR        = os.path.join(BASE_DIR, "json")
PROMPTS_DIR     = os.path.join(BASE_DIR, "prompts")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
PROMPT_FILE     = os.path.join(PROMPTS_DIR, "prompt.txt")
GROQ_MODELS_DIR = os.path.join(BASE_DIR, "groq_modelos")
PRONT_SQL_FILE  = os.path.join(SQL_DIR, "select_prontuarios.sql")
EXAMS_SQL_FILE  = os.path.join(SQL_DIR, "select_resultado_exames.sql")

POSTOS = ["A", "N", "X", "Y", "B", "R", "P", "C", "D", "G", "I", "J", "M"]

# ---------------- Utils ----------------
def ensure_dirs():
    for d in (SQL_DIR, JSON_DIR, PROMPTS_DIR, REPORTS_DIR, GROQ_MODELS_DIR):
        os.makedirs(d, exist_ok=True)

def purge_old_files(dirpath: str, older_than_hours: int = 1):
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
    import unicodedata as _u
    return "".join(c for c in _u.normalize("NFD", s or "") if _u.category(c) != "Mn")

# ---------------- Conexão DB ----------------
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

# ---------- Load SQL (único para todos os postos) ----------
def _read_file_any_encoding(path: str) -> str:
    encs = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def load_prontuario_sql() -> str:
    if not os.path.isfile(PRONT_SQL_FILE):
        # fallback seguro
        return (
            "select "
            "p.idprontuario, p.idcliente, p.iddependente, p.matricula, p.idendereco, "
            "p.paciente, p.idade, p.datanascimento, p.peso, p.altura, "
            "p.parterialinicio, p.parterialfim, p.doencascronicas, p.medicaoanterior, "
            "p.queixa, p.conduta, p.desativado, p.datainicioconsulta, p.datafimconsulta, "
            "p.idmedico, p.observacao, p.informacao, p.temperatura, p.bpm, "
            "p.idespecialidade, p.especialidade, p.nomesocial, p.pacienteatendido, "
            "m.Nome as profissional_atendente "
            "from cad_prontuario p "
            "left join cad_medico m on m.idMedico = p.idmedico "
            "where p.paciente = :paciente and p.DataNascimento = :nasc "
            "and p.DataInicioConsulta >= :dt_ini "
            "and p.DataInicioConsulta is not null and p.DataFimConsulta is not null "
            "and isNull(falta,0) = 0 and p.desativado = 0"
        )
    sql = _read_file_any_encoding(PRONT_SQL_FILE).strip()
    # Corrige "?" -> parâmetros nomeados (na ordem: paciente, nasc, dt_ini)
    if "?" in sql and ":paciente" not in sql:
        sql = sql.replace("?", ":paciente", 1).replace("?", ":nasc", 1).replace("?", ":dt_ini", 1)
    # pequenas correções
    sql = re.sub(r"\band\s+and\b", "and", sql, flags=re.IGNORECASE)
    return sql

def load_exam_sql() -> str:
    if not os.path.isfile(EXAMS_SQL_FILE):
        return (
            "select p.datanascimento, lsri.* "
            "from vw_Cad_LancamentoServicoResultadoItem lsri "
            "left join vw_cad_paciente p on p.matricula = lsri.matricula and lsri.paciente = p.nome "
            "where (LEN(lsri.exameResultado) > 0) "
            "and lsri.paciente = :paciente and p.DataNascimento = :nasc and lsri.dataliberado >= :dt_ini "
            "and lsri.desativado = 0 "
            "order by lsri.dataliberado desc"
        )
    sql = _read_file_any_encoding(EXAMS_SQL_FILE).strip()
    if "?" in sql and ":paciente" not in sql:
        sql = sql.replace("?", ":paciente", 1).replace("?", ":nasc", 1).replace("?", ":dt_ini", 1)
    sql = re.sub(r"\bmatriculal\b", "matricula", sql, flags=re.IGNORECASE)
    return sql

# ---------- Queries ----------
def wrap_with_filters(base_sql: str) -> text:
    return text(base_sql)

def query_posto(label: str, odbc_conn_str: str, paciente: str, nasc_date: date, dt_ini: date, use_like=False) -> pd.DataFrame:
    print(f"[{label}] Conectando...")
    base_sql = load_prontuario_sql()
    if use_like:
        base_sql = re.sub(
            r"p\.paciente\s*=\s*:paciente",
            "LTRIM(RTRIM(p.paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI "
            "LIKE ('%' + LTRIM(RTRIM(:paciente)) + '%') COLLATE SQL_Latin1_General_CP1_CI_AI",
            base_sql, flags=re.IGNORECASE,
        )
    else:
        base_sql = re.sub(
            r"p\.paciente\s*=\s*:paciente",
            "LTRIM(RTRIM(p.paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI = "
            "LTRIM(RTRIM(:paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI",
            base_sql, flags=re.IGNORECASE,
        )
    sql = wrap_with_filters(base_sql)
    params = {"paciente": paciente.strip(), "nasc": nasc_date, "dt_ini": dt_ini}
    try:
        engine = make_engine(odbc_conn_str)
        with engine.begin() as con:
            df = pd.read_sql(sql, con=con, params=params)
        print(f"[{label}] {('Nenhum registro' if df.empty else str(len(df))+' registro(s)')}")
        return df
    except Exception as e:
        print(f"[{label}] ERRO: {e}")
        return pd.DataFrame()

def query_exams_posto(label: str, odbc_conn_str: str, paciente: str, nasc_date: date, dt_ini: date) -> pd.DataFrame:
    sql_txt = load_exam_sql()
    try:
        engine = make_engine(odbc_conn_str)
        with engine.begin() as con:
            df = pd.read_sql(text(sql_txt), con=con, params={"paciente": paciente.strip(), "nasc": nasc_date, "dt_ini": dt_ini})
        if not df.empty:
            df.insert(0, "posto", label)
        print(f"[{label}-EX] {('Nenhum resultado' if df.empty else str(len(df))+' resultado(s)')}")
        return df
    except Exception as e:
        msg = str(e)
        if "916" in msg or "no contexto de segurança atual" in msg or "contexto de segurança atual" in msg:
            print(f"[{label}-EX] Sem permissão para ler exames (916). Ignorado.")
        else:
            print(f"[{label}-EX] ERRO: {msg.splitlines()[0]}")
        return pd.DataFrame()

# -------- CSV de exames (fallback) --------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.strip() for c in df2.columns]
    return df2

def _parse_date_any(x):
    if pd.isna(x): return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
        try: return datetime.strptime(str(x)[:10], fmt).date()
        except: pass
    try:
        v = pd.to_datetime(x, errors="coerce")
        return None if pd.isna(v) else v.date()
    except:
        return None

def load_exams_from_csvs(nome: str, nasc_date: date, dt_ini: date) -> pd.DataFrame:
    """
    Lê qualquer arquivo ./reports/exames*.csv (inclusive 'exames.csv'),
    filtra por paciente + nascimento + data >= dt_ini (se disponível),
    retorna DF com coluna 'posto'='CSV'.
    """
    paths = sorted(glob.glob(os.path.join(REPORTS_DIR, "exames*.csv")))
    if not paths:
        return pd.DataFrame()
    frames = []
    for pth in paths:
        try:
            df = pd.read_csv(pth, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(pth, dtype=str, encoding="latin-1")
        df = _normalize_cols(df)
        cols_lower = {c.lower(): c for c in df.columns}
        c_pac = cols_lower.get("paciente") or cols_lower.get("nome") or cols_lower.get("nomepaciente") or cols_lower.get("nm_paciente")
        c_nasc = cols_lower.get("datanascimento") or cols_lower.get("data_nascimento") or cols_lower.get("dt_nasc") or cols_lower.get("nascimento")
        c_data = cols_lower.get("dataliberado") or cols_lower.get("data") or cols_lower.get("datacoleta") or cols_lower.get("datalancamento")
        if not (c_pac and c_nasc):
            continue
        df["_pac"]  = df[c_pac].astype(str).str.strip()
        df["_nasc"] = df[c_nasc].map(_parse_date_any)
        mask = (df["_pac"].str.casefold() == nome.strip().casefold()) & (df["_nasc"] == nasc_date)
        if c_data:
            df["_dt"] = df[c_data].map(_parse_date_any)
            mask = mask & (df["_dt"].notna() & (df["_dt"] >= dt_ini))
        df_sel = df.loc[mask].copy()
        if not df_sel.empty:
            df_sel.insert(0, "posto", "CSV")
            frames.append(df_sel.drop(columns=["_pac","_nasc","_dt"], errors="ignore"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ---------------- JSON helpers ----------------
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

# ---------------- Filtros / bloco1 ----------------
def _no_show_row(row: dict) -> bool:
    txt = " ".join([str(row.get(k, "")) for k in ("queixa", "observacao", "conduta")])
    txt_norm = strip_accents(txt).lower()
    return ("nao compareceu" in txt_norm) and ("chamad" in txt_norm)

def build_last10(df_all: pd.DataFrame) -> pd.DataFrame:
    """Seleciona os 10 últimos atendimentos (sem gerar 'resumo' por linha)."""
    df = df_all.copy()
    df["_dt_ini"] = pd.to_datetime(df["datainicioconsulta"], errors="coerce")
    for c in ["idprontuario","posto","_dt_ini","especialidade","queixa","observacao","conduta"]:
        if c not in df.columns:
            df[c] = None
    df = (
        df.loc[df["_dt_ini"].notna(),
               ["idprontuario","posto","_dt_ini","especialidade","queixa","observacao","conduta"]]
          .sort_values("_dt_ini", ascending=False)
          .head(10)
          .copy()
    )
    df["data"] = df["_dt_ini"].dt.strftime("%d/%m/%Y")
    return df

def narrate_last10(df_b1: pd.DataFrame) -> str:
    """
    Gera título + uma linha por atendimento no formato:
    Em DD/MM/AAAA, na <especialidade>, registrou-se: <trecho>.
    """
    titulo = "RESUMO DOS ÚLTIMOS ATENDIMENTOS"
    if df_b1 is None or df_b1.empty:
        return f"{titulo}\n\n"
    def _short(*vals, maxlen=280):
        txt = " ".join([clean_text(v) for v in vals if v])
        txt = re.sub(r"\s+", " ", txt).strip(" .")
        return (txt[:maxlen] + "…") if len(txt) > maxlen else txt
    linhas = []
    for _, r in df_b1.iterrows():
        data = r.get("data") or ""
        esp  = clean_text(r.get("especialidade") or "").lower()
        que  = clean_text(r.get("queixa") or "")
        obs  = clean_text(r.get("observacao") or "")
        con  = clean_text(r.get("conduta") or "")
        trecho = _short(que, obs, con)
        if esp and trecho:
            linhas.append(f"Em {data}, na {esp}, registrou-se: {trecho}.")
        elif esp:
            linhas.append(f"Em {data}, na {esp}, atendimento sem detalhes textuais relevantes.")
        else:
            linhas.append(f"Em {data}, atendimento sem detalhes textuais relevantes.")
    texto = "\n".join(linhas)
    return f"{titulo}\n\n{texto}"

# ---------------- Prompt ----------------
def load_user_prompt() -> str:
    if not os.path.isfile(PROMPT_FILE):
        return (
            "# STRICT_EVIDENCE_PROMPT (fallback)\n"
            "Devolva SOMENTE JSON com chaves: bloco2 (lista), bloco3 (lista), "
            "bloco4 {observacao,conduta}, rodape (lista). Não invente fatos."
        )
    return _read_file_any_encoding(PROMPT_FILE).strip()

def _compact_exam_row(r: Dict[str, Any]) -> Dict[str, Any]:
    keys = {k.lower(): k for k in r.keys()}
    def g(*opts):
        for o in opts:
            if o.lower() in keys:
                return r[keys[o.lower()]]
        return None
    def trunc(s, n=200):
        s = clean_text(s)
        return (s[:n] + "…") if s and len(s) > n else s
    return {
        "posto": r.get("posto"),
        "data": g("dataliberado","dataresultado","data","datacoleta","datalancamento"),
        "grupo": g("grupo"),
        "servico": g("servico"),
        "exame": g("exame","exameitem","examedescricao"),
        "resultado": trunc(g("exameresultado","resultado")),
        "referencia": trunc(g("referencia","valorreferencia","intervaloreferencia")),
        "unidade": g("unidade","unidademedida"),
        "observacao": trunc(g("observacao","observacoes")),
    }

def build_prompt_chunk(queixa_atual: str,
                       historicos_chunk: List[Dict[str, Any]],
                       user_prompt: str,
                       part_idx: int,
                       part_total: int,
                       exams_compact: List[Dict[str, Any]],
                       bloco1_observacao: str,
                       bloco1_narrativa: str) -> str:
    # Bloco 1 vem pronto e NÃO deve ser alterado; não enviamos array de "últimos atendimentos".
    hist_json = json.dumps(historicos_chunk, ensure_ascii=False)
    exams_json = json.dumps(exams_compact, ensure_ascii=False)

    header = (
        f"QUEIXA_ATUAL:\n{queixa_atual}\n\n"
        f"BLOCO1_TEXTO (NÃO ALTERAR, já é a narrativa final):\n{bloco1_narrativa or '—'}\n\n"
        f"BLOCO1_OBSERVACAO:\n{bloco1_observacao or '—'}\n\n"
        f"EXAMES_JSON (compacto, considerar em bloco2/3/4):\n{exams_json}\n\n"
        f"HISTORICO_JSON (PARTE {part_idx}/{part_total}):\n{hist_json}\n\n"
        f"Instruções: Responda SOMENTE JSON com as chaves: "
        f"bloco2 (lista), bloco3 (lista), bloco4 {{observacao,conduta}}, rodape (lista)."
    )
    return header + ("\n\n" + (user_prompt or ""))

# ---------------- Groq models (.ini) ----------------
def _list_groq_model_configs() -> List[Dict[str, Any]]:
    items = {}
    if not os.path.isdir(GROQ_MODELS_DIR):
        return []
    for fn in os.listdir(GROQ_MODELS_DIR):
        if not re.match(r"^\d+-.*\.(ini|txt)$", fn, re.I):
            continue
        m = re.match(r"^(\d+)-", fn)
        order = int(m.group(1)) if m else 9999
        path = os.path.join(GROQ_MODELS_DIR, fn)
        cfg = {
            "order": order,
            "path": path,
            "model": None,
            "temperature": 0.1,
            "top_p": 1.0,
            "max_tokens": 1800,
            "reasoning_effort": "",
            "json_mode": True,
            "stop": None,
            "name": os.path.splitext(fn)[0],
        }
        try:
            txt = _read_file_any_encoding(path).replace("\r\n", "\n")
            parser = ConfigParser(inline_comment_prefixes=("#", ";"))
            parser.read_string(txt)
            if parser.has_section("groq"):
                g = parser["groq"]
                def getf(key, default=None):
                    return g.get(key, fallback=default)
                model = (getf("model","") or "").strip().strip('"').strip("'")
                if model: cfg["model"] = model
                if getf("temperature") is not None:
                    try: cfg["temperature"] = float(getf("temperature"))
                    except: pass
                if getf("top_p") is not None:
                    try: cfg["top_p"] = float(getf("top_p"))
                    except: pass
                if getf("max_tokens") is not None:
                    try: cfg["max_tokens"] = int(getf("max_tokens"))
                    except: pass
                eff = (getf("reasoning_effort","") or "").strip().strip('"').strip("'")
                cfg["reasoning_effort"] = eff
                jm = (str(getf("json_mode","true"))).lower()
                cfg["json_mode"] = jm in {"1","true","yes","y"}
                stop_raw = getf("stop","")
                if stop_raw:
                    try:
                        cfg["stop"] = json.loads(stop_raw)
                    except Exception:
                        cfg["stop"] = [s.strip() for s in re.split(r"[;,]\s*", stop_raw.strip(" []")) if s.strip()]
        except Exception:
            pass
        if cfg["model"]:
            items.setdefault(order, cfg)
    return [items[k] for k in sorted(items.keys())]

# ---------------- Groq HTTP client ----------------
def _request_with_retries(func, label: str, attempts: int = 3, backoff: float = 1.5) -> Optional[str]:
    import random
    for i in range(attempts):
        try:
            out = func()
            if out and out.strip():
                return out
        except requests.HTTPError as e:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text if e.response is not None else ""
            print(f"[{label}] HTTP {getattr(e.response,'status_code',None)}: {str(body)}")
            try:
                if isinstance(body, dict) and body.get("error", {}).get("code") == "json_validate_failed" and "gpt-oss-120b" in label.lower():
                    return None
            except Exception:
                pass
            retry_after = 0.0
            if e.response is not None:
                ra = e.response.headers.get("Retry-After")
                try:
                    retry_after = float(ra)
                except Exception:
                    retry_after = 0.0
            if i < attempts - 1:
                delay = max(retry_after, (backoff ** i)) + random.uniform(0,0.4)
                print(f"[{label}] retry em {delay:.1f}s...")
                time.sleep(delay)
                continue
            return None
        except Exception as e:
            print(f"[{label}] erro: {e}")
        if i < attempts - 1:
            delay = (backoff ** i)
            print(f"[{label}] vazio/sem resposta. retry em {delay:.1f}s...")
            time.sleep(delay)
    return None

def groq_chat_json(prompt: str, cfg: Dict[str, Any]) -> Optional[str]:
    api_key = _env("GROQ_API_KEY", "")
    if not api_key:
        print("[GROQ] falta GROQ_API_KEY.")
        return None

    model_name = cfg["model"]
    max_toks = int(cfg.get("max_tokens", 1800))
    if "gpt-oss-120b" in (model_name or ""):
        max_toks = max(max_toks, 3500)
        cfg["temperature"] = 0.0

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Você é um assistente clínico objetivo. Responda em pt-BR."},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.get("temperature", 0.1),
        "top_p": cfg.get("top_p", 1),
        "max_tokens": max_toks,
        "stream": False,
    }
    if cfg.get("json_mode", True):
        body["response_format"] = {"type": "json_object"}
    if cfg.get("stop"):
        body["stop"] = cfg["stop"]
    eff = (cfg.get("reasoning_effort") or "").strip()
    if eff:
        body["reasoning"] = {"effort": eff}

    label = f"GROQ-{model_name}"
    print(f"[GROQ] tentando modelo: {model_name}")

    def _do():
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            try:
                print(f"[{label}] HTTP {resp.status_code}: {resp.json()}")
            except Exception:
                print(f"[{label}] HTTP {resp.status_code}: {resp.text[:800]}")
            resp.raise_for_status()
        data = resp.json()
        choices = (data or {}).get("choices") or []
        if not choices or not choices[0].get("message", {}).get("content"):
            raise RuntimeError("groq_empty_or_malformed")
        return choices[0]["message"]["content"].strip()

    return _request_with_retries(_do, label=label, attempts=3, backoff=1.5)

# -------- JSON resiliente --------
def _safe_json_loads(txt: str) -> Optional[dict]:
    if not txt:
        return None
    s = txt.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).rstrip("`").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            return None
    return None

# ---------------- Map → Reduce (chunking) ----------------
def call_groq_map_reduce(queixa_atual: str,
                         historicos: List[Dict[str, Any]],
                         model_cfgs: List[Dict[str, Any]],
                         user_prompt: str,
                         exams_compact: List[Dict[str, Any]],
                         bloco1_observacao: str,
                         bloco1_narrativa: str) -> Tuple[Optional[str], Dict[str, Any]]:
    if not model_cfgs:
        return (None, {})

    for cfg in model_cfgs:
        is_120b = "gpt-oss-120b" in (cfg.get("model") or "")
        CHUNK_MAX_ITEMS = 60 if is_120b else 120
        parts = [historicos[i:i+CHUNK_MAX_ITEMS] for i in range(0, len(historicos), CHUNK_MAX_ITEMS)]
        total_parts = max(1, len(parts))

        agg_bloco2: List[str] = []
        agg_bloco3: List[str] = []
        agg_obs, agg_cond = "", ""
        agg_rodape: List[str] = []

        ok_all = True
        for idx, chunk in enumerate(parts, start=1):
            prompt = build_prompt_chunk(
                queixa_atual, chunk, user_prompt, idx, total_parts,
                exams_compact, bloco1_observacao, bloco1_narrativa
            )
            out = groq_chat_json(prompt, cfg)
            if not out:
                ok_all = False
                break
            parsed = None
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = _safe_json_loads(out)
            if not isinstance(parsed, dict):
                ok_all = False
                break

            if isinstance(parsed.get("bloco2"), list):
                agg_bloco2 += [str(x) for x in parsed["bloco2"]]
            if isinstance(parsed.get("bloco3"), list):
                agg_bloco3 += [str(x) for x in parsed["bloco3"]]
            if isinstance(parsed.get("bloco4"), dict):
                o = str(parsed["bloco4"].get("observacao","")).strip()
                c = str(parsed["bloco4"].get("conduta","")).strip()
                if o: agg_obs = o
                if c: agg_cond = c
            if isinstance(parsed.get("rodape"), list):
                agg_rodape += [str(x) for x in parsed["rodape"]]

        if not ok_all:
            continue

        def _dedup(seq: List[str]) -> List[str]:
            seen, out = set(), []
            for s in [clean_text(x) for x in seq if clean_text(x)]:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out

        result = {
            "bloco2": _dedup(agg_bloco2)[:30],
            "bloco3": _dedup(agg_bloco3)[:30],
            "bloco4": {"observacao": agg_obs, "conduta": agg_cond},
            "rodape": _dedup(agg_rodape)[:30] or ["NENHUM"]
        }
        return (cfg["model"], result)

    return (None, {})

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Busca de prontuários multi-postos + exames + análise Groq (JSON).")
    p.add_argument("-n", "--nome", help="Nome completo do paciente")
    p.add_argument("-d", "--nascimento", help="Data de nascimento dd/mm/yyyy")
    p.add_argument("-q", "--queixa", help="Queixa atual")
    p.add_argument("--like", action="store_true", help="Se não achar por igualdade, tenta LIKE automaticamente (consultas)")
    return p.parse_args()

def _ask_date(msg: str) -> date:
    while True:
        s = input(msg).strip()
        try:
            return datetime.strptime(s, "%d/%m/%Y").date()
        except ValueError:
            print("Data inválida. Use dd/mm/yyyy.")

def main():
    ensure_dirs()
    purge_old_files(JSON_DIR, 1)
    purge_old_files(REPORTS_DIR, 1)

    args = parse_args()
    nome = args.nome or input("1) Nome completo do paciente: ").strip()
    data_nasc_str = args.nascimento or input("2) Data de nascimento (dd/mm/yyyy): ").strip()
    queixa = args.queixa or input("3) Queixa atual do paciente: ").strip()
    dt_ini = _ask_date("4) Buscar desde (dd/mm/yyyy): ")

    try:
        nasc_date = datetime.strptime(data_nasc_str, "%d/%m/%Y").date()
    except ValueError:
        print(json.dumps({"ok": False, "erro": "Data inválida. Use dd/mm/yyyy."}, ensure_ascii=False))
        return

    conns = build_conns_from_env()
    if not conns:
        print(json.dumps({"ok": False, "erro": "Nenhum posto configurado no .env."}, ensure_ascii=False))
        return

    # -------- Consultas (por posto) --------
    def _run_consultas(dt_inicio: date, like_flag: bool) -> pd.DataFrame:
        frames = []
        for lbl, conn_str in conns.items():
            df = query_posto(lbl, conn_str, nome, nasc_date, dt_inicio, use_like=like_flag)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df_all = _run_consultas(dt_ini, like_flag=False)
    if df_all.empty and (args.like or True):
        print("Nenhum dado com igualdade exata. Tentando busca parcial (LIKE)...")
        df_all = _run_consultas(dt_ini, like_flag=True)

    # -------- Exames (SQL único, igualdade exata) --------
    frames_ex = []
    for lbl, conn_str in conns.items():
        df_ex = query_exams_posto(lbl, conn_str, nome, nasc_date, dt_ini)
        if not df_ex.empty:
            frames_ex.append(df_ex)
    df_exams_all = pd.concat(frames_ex, ignore_index=True) if frames_ex else pd.DataFrame()

    # Fallback CSV se DB não devolver nada
    if df_exams_all.empty:
        df_csv = load_exams_from_csvs(nome, nasc_date, dt_ini)
        if not df_csv.empty:
            print(f"[EX-CSV] {len(df_csv)} linha(s) carregadas de CSV.")
            df_exams_all = df_csv

    # -------- Bloco 1: 10 últimos e narrativa única --------
    if df_all.empty:
        df_b1 = pd.DataFrame(columns=["idprontuario","posto","data","especialidade","queixa","observacao","conduta"])
    else:
        df_b1 = build_last10(df_all)

    bloco1_narrativa = narrate_last10(df_b1)

    # ------ HISTÓRICO p/ IA (consultas) ------
    historicos = []
    if not df_all.empty:
        df_all.columns = [str(c).strip().lower() for c in df_all.columns]
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente","datainicioconsulta"]:
            if col not in df_all.columns:
                df_all[col] = None
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente"]:
            df_all[col] = df_all[col].map(clean_text)

        df_hist = df_all.copy()
        mask_send = []
        for _, row in df_hist.iterrows():
            mask_send.append(not _no_show_row(row.to_dict()))
        df_hist_send = df_hist.loc[mask_send].reset_index(drop=True)

        def _is_blank_series(s: pd.Series) -> pd.Series:
            if s is None:
                return pd.Series([True] * 0)
            return s.isna() | s.astype(str).str.strip().eq("") | s.astype(str).str.lower().eq("none")
        q_blank = _is_blank_series(df_hist_send.get("queixa", pd.Series([None] * len(df_hist_send))))
        c_blank = _is_blank_series(df_hist_send.get("conduta", pd.Series([None] * len(df_hist_send))))
        df_hist_send = df_hist_send.loc[~(q_blank & c_blank)].reset_index(drop=True)

        historicos = sanitize_for_json(df_hist_send).to_dict(orient="records")

    # ------ EXAMES: compacta p/ prompt, exporta CSV (se veio do DB), entrega JSON ------
    exames_list = []
    exames_csv_path = None
    exams_compact_for_prompt: List[Dict[str, Any]] = []
    if not df_exams_all.empty:
        if frames_ex:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            exames_csv_path = os.path.join(REPORTS_DIR, f"exames_{ts}.csv")
            df_exams_all.to_csv(exames_csv_path, index=False, encoding="utf-8-sig")
        exames_list = sanitize_for_json(df_exams_all).to_dict(orient="records")
        for r in exames_list[:100]:
            exams_compact_for_prompt.append(_compact_exam_row(r))

    # salvar JSON bruto (auditoria)
    payload_full = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "data_inicial_utilizada": dt_ini.isoformat(),
            "criterio_tempo": "desde_data_informada",
            "gerado_em": datetime.now().isoformat(),
            "registros_total_consultas": int(len(df_all)) if not isinstance(df_all, list) else 0,
            "registros_total_exames": int(len(df_exams_all)) if not isinstance(df_exams_all, list) else 0,
        },
        "consultas_amostra": sanitize_for_json(df_all).to_dict(orient="records") if not df_all.empty else [],
        "exames_amostra": exames_list,
    }
    json_path = save_json(payload_full)
    print(f"JSON: {os.path.basename(json_path)} (será limpo em ~1h)")

    # ---------------- IA (Groq) ----------------
    prompt_txt = load_user_prompt()
    model_cfgs = _list_groq_model_configs()
    bloco1_observacao = ""  # use se quiser sinalizar alguma observação do bloco 1

    model_used, result = call_groq_map_reduce(
        queixa, historicos, model_cfgs, prompt_txt,
        exams_compact_for_prompt, bloco1_observacao, bloco1_narrativa
    )

    if not result:
        result = {
            "bloco2": ["não houve resposta da IA - Falha de comunicação. Tem internet?"],
            "bloco3": ["não houve resposta da IA - Falha de comunicação. Tem internet?"],
            "bloco4": {"observacao": "não houve resposta da IA - Falha de comunicação. Tem internet?",
                       "conduta": "não houve resposta da IA - Falha de comunicação. Tem internet?"},
            "rodape": ["Itens sem evidência textual explícita: NENHUM"]
        }

    # ---------------- Saída FINAL ----------------
    output = {
        "ok": True,
        "paciente": nome,
        "data_nascimento": nasc_date.isoformat(),
        "queixa_atual": queixa,
        "registros_total": int(len(df_all)) if not df_all.empty else 0,
        "meses_busca": None,
        "data_inicial": dt_ini.isoformat(),
        "bloco1_observacao": bloco1_observacao,
        "BLOCO1_RESUMO": (bloco1_narrativa or "")[:4000],  # mantém título + linhas
        "provider_mode": "json" if model_used else "fail",
        "provider_used": model_used,
        # REMOVIDO: "ULTIMOS ATENDIMENTOS"
        "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": result.get("bloco2", []),
        "DIAGNOSTICO DIFERENCIAL": result.get("bloco3", []),
        "SUGESTAO CAMPOS OBS E CONDUTA": result.get("bloco4", {"observacao":"", "conduta":""}),
        "EXAMES_RESULTADOS": exames_list,
        "exames_csv": exames_csv_path,
        "rodape": result.get("rodape", ["NENHUM"])
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

# ---------------- Entry ----------------
if __name__ == "__main__":
    main()
