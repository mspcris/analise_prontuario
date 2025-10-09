# -*- coding: utf-8 -*-
"""
project.py — Busca prontuários multi-postos e exames de sangue (120 dias),
gera Bloco 1 local (fallback) e produz Blocos via GROQ (texto livre com tags).
Sem regras adicionais no código: o comportamento segue o que está em prompts/prompt.txt.

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv requests python-dateutil
"""

import os, re, json, uuid, time, argparse, unicodedata
from datetime import datetime, date, timedelta
from urllib.parse import quote_plus
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from configparser import ConfigParser
import requests
from dateutil.relativedelta import relativedelta

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

# Lookback de exames de sangue (regra solicitada)
EXAMS_BLOOD_LOOKBACK_DAYS = 120

# ---------------- Utils ----------------
def ensure_dirs():
    for d in (SQL_DIR, JSON_DIR, PROMPTS_DIR, REPORTS_DIR, GROQ_MODELS_DIR):
        os.makedirs(d, exist_ok=True)

def purge_old_files(dirpath: str, older_than_hours: int = 24):
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
    if isinstance(x, (np.integer,)):  return int(x)
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
    if s is None: return ""
    s = str(s).replace("\r\n"," ").replace("\n"," ").replace("\r"," ")
    return re.sub(r"\s+"," ",s).strip()

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn")

def safe_concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    cleaned = []
    for df in frames:
        if df is None or df.empty:
            continue
        df2 = df.loc[:, ~df.isna().all(axis=0)].copy()
        if not df2.empty:
            cleaned.append(df2)
    return pd.concat(cleaned, ignore_index=True, sort=False) if cleaned else pd.DataFrame()

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

# ---------- Load SQL ----------
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
        return (
            "select "
            "p.idprontuario, p.matricula, p.paciente, p.datanascimento, p.queixa, p.observacao, p.conduta, "
            "p.datainicioconsulta, p.datafimconsulta, p.especialidade, p.idmedico, "
            "m.Nome as profissional_atendente, p.desativado "
            "from cad_prontuario p "
            "left join cad_medico m on m.idMedico = p.idmedico "
            "where LTRIM(RTRIM(p.paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI = "
            "LTRIM(RTRIM(:paciente)) COLLATE SQL_Latin1_General_CP1_CI_AI "
            "and p.DataNascimento = :nasc "
            "and p.DataInicioConsulta >= :dt_ini "
            "and p.DataInicioConsulta is not null and p.DataFimConsulta is not null "
            "and isNull(falta,0) = 0 and p.desativado = 0"
        )
    sql = _read_file_any_encoding(PRONT_SQL_FILE).strip()
    if "?" in sql and ":paciente" not in sql:
        sql = sql.replace("?", ":paciente", 1).replace("?", ":nasc", 1).replace("?", ":dt_ini", 1)
    sql = re.sub(r"\band\s+and\b", "and", sql, flags=re.IGNORECASE)
    return sql

def load_exam_sql() -> str:
    # Usa o select_resultado_exames.sql (UTF-8) que você alterou (idResultado, idlancamentoservico etc).
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
        print(f"[{label}] ERRO: {str(e).splitlines()[0]}")
        return pd.DataFrame()

def query_exams_all_posts(conns: Dict[str,str], paciente: str, nasc_date: date, dt_ini_exams: date) -> pd.DataFrame:
    sql_txt = load_exam_sql()
    frames = []
    for lbl, conn_str in conns.items():
        try:
            engine = make_engine(conn_str)
            with engine.begin() as con:
                df = pd.read_sql(text(sql_txt), con=con, params={"paciente": paciente.strip(), "nasc": nasc_date, "dt_ini": dt_ini_exams})
            if not df.empty:
                df.insert(0, "posto", lbl)
                print(f"[{lbl}-EX] {len(df)} resultado(s)")
                frames.append(df)
            else:
                print(f"[{lbl}-EX] Nenhum resultado")
        except Exception as e:
            msg = str(e)
            if "8114" in msg:
                print(f"[{lbl}-EX] ERRO 8114 (varchar→bigint) — verifique CAST simétrico no SQL do laboratório.")
            elif "916" in msg or "no contexto de segurança atual" in msg or "contexto de segurança atual" in msg:
                print(f"[{lbl}-EX] Sem permissão para ler exames (916). Ignorado.")
            else:
                print(f"[{lbl}-EX] ERRO: {msg.splitlines()[0]}")
    return safe_concat(frames)

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

# ---------------- Regras de EXAMES DE SANGUE ----------------
def _is_blood_exam_row(r: Dict[str, Any]) -> bool:
    txt = " ".join([
        str(r.get(k,"")) for k in ["material","grupo","setor","servico","exame","item","servicoexamematerial"]
        if k in r
    ])
    t = strip_accents(clean_text(txt)).lower()
    if "sang" in t:  # "sangue", "sanguinea"
        return True
    keys = ["hemograma","hemat", "bioquim", "hormon", "serol", "imuno", "coagul", "ferro", "glic", "perfil lipid", "lipid"]
    return any(k in t for k in keys)

def _blood_lookback_date() -> date:
    return (date.today() - timedelta(days=EXAMS_BLOOD_LOOKBACK_DAYS))

# ---------------- Bloco 1 (fallback Py) ----------------
def build_last10(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["_dt_ini"] = pd.to_datetime(df.get("datainicioconsulta"), errors="coerce")
    need_cols = ["idprontuario","posto","_dt_ini","especialidade","queixa","observacao","conduta"]
    for c in need_cols:
        if c not in df.columns: df[c] = None
    df = (
        df.loc[df["_dt_ini"].notna(), need_cols]
          .sort_values("_dt_ini", ascending=False)
          .head(10)
          .copy()
    )
    df["data_fmt"] = df["_dt_ini"].dt.strftime("%d/%m/%Y")
    def _mk_resumo(r):
        partes = [clean_text(r["queixa"]), clean_text(r["observacao"]), clean_text(r["conduta"])]
        partes = [p for p in partes if p]
        return (". ".join(partes).strip(" .") + ".") if partes else ""
    df["linha_fmt"] = df.apply(lambda r: f"Em {r['data_fmt']}, na {clean_text(r.get('especialidade') or '').lower()}, registrou-se: {_mk_resumo(r)}".strip(), axis=1)
    return df

def summarize_block1_lines(lines: List[str], maxlen=1000) -> str:
    head = "RESUMO DOS ÚLTIMOS ATENDIMENTOS (Py)\n\n"
    if not lines:
        return head.strip()
    full = head + "\n".join([l for l in lines if l])
    if len(full) <= maxlen:
        return full
    budget = maxlen - len(head)
    out_lines, acc = [], 0
    for l in lines:
        L = len(l) + 1
        if acc + L > budget:
            break
        out_lines.append(l); acc += L
    if not out_lines:
        return (head + lines[0][:budget]).rstrip()
    return (head + "\n".join(out_lines)).rstrip()

def build_block1_list_py(df_all: pd.DataFrame) -> List[str]:
    if df_all.empty: return []
    df = df_all.copy()
    df["_dt"] = pd.to_datetime(df.get("datainicioconsulta"), errors="coerce")
    df = df.loc[df["_dt"].notna()].sort_values("_dt", ascending=False).head(5)
    out = []
    for _, r in df.iterrows():
        dt_vis = r["_dt"].strftime("%d/%m/%Y")
        dt_iso = r["_dt"].strftime("%Y-%m-%dT%H:%M:%S")
        rid = r.get("idprontuario")
        posto = r.get("posto")
        resumo = " ".join([clean_text(r.get("queixa")), clean_text(r.get("observacao")), clean_text(r.get("conduta"))]).strip()
        resumo = re.sub(r"\s+", " ", resumo)
        out.append(f"{dt_vis}, {rid}, {resumo} (Posto {posto}, id {rid}, data {dt_iso})")
    return out

# ---------------- Prompt loader ----------------
def load_user_prompt() -> str:
    # Apenas lê o prompt do arquivo, sem adicionar regras no código.
    return _read_file_any_encoding(PROMPT_FILE).strip() if os.path.isfile(PROMPT_FILE) else ""

# ---------------- GROQ models (.ini) ----------------
def _list_groq_model_configs() -> List[Dict[str, Any]]:
    # Se houver configs no diretório, respeita a ordem numérica; senão, cria defaults (120B primeiro).
    items_ordered: List[Dict[str, Any]] = []
    if os.path.isdir(GROQ_MODELS_DIR):
        temp = {}
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
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 700,
                "json_mode": False,  # 120B: texto livre
                "stop": None,
                "name": os.path.splitext(fn)[0],
            }
            try:
                txt = _read_file_any_encoding(path).replace("\r\n", "\n")
                parser = ConfigParser(inline_comment_prefixes=("#", ";"))
                parser.read_string(txt)
                if parser.has_section("groq"):
                    g = parser["groq"]
                    model = (g.get("model","") or "").strip().strip('"').strip("'")
                    if model: cfg["model"] = model
                    try: cfg["temperature"] = float(g.get("temperature", cfg["temperature"]))
                    except: pass
                    try: cfg["top_p"] = float(g.get("top_p", cfg["top_p"]))
                    except: pass
                    try: cfg["max_tokens"] = int(g.get("max_tokens", cfg["max_tokens"]))
                    except: pass
                    jm = (str(g.get("json_mode","false"))).lower()
                    cfg["json_mode"] = jm in {"1","true","yes","y"}
                    stop_raw = g.get("stop","")
                    if stop_raw:
                        try: cfg["stop"] = json.loads(stop_raw)
                        except: cfg["stop"] = [s.strip() for s in re.split(r"[;,]\s*", stop_raw.strip(" []")) if s.strip()]
            except Exception:
                pass
            if cfg["model"]:
                temp.setdefault(order, cfg)
        items_ordered = [temp[k] for k in sorted(temp.keys())]
    if not items_ordered:
        # Defaults: 120B (texto livre), depois Llama 70B (ainda texto com tags)
        items_ordered = [
            {"order": 1, "model": "openai/gpt-oss-120b", "temperature": 0.0, "top_p": 1.0, "max_tokens": 700, "json_mode": False, "stop": None, "name": "default-120b"},
            {"order": 2, "model": "llama-3.3-70b-versatile", "temperature": 0.2, "top_p": 0.9, "max_tokens": 900, "json_mode": False, "stop": None, "name": "fallback-70b"},
        ]
    return items_ordered

# ---------------- Groq HTTP client (texto livre) ----------------
def groq_chat_text(prompt: str, cfg: Dict[str, Any]) -> Optional[str]:
    api_key = _env("GROQ_API_KEY", "")
    if not api_key:
        print("[GROQ] falta GROQ_API_KEY.")
        return None
    model_name = cfg["model"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}  # textos e tags 100% do prompt.txt
        ],
        "temperature": float(cfg.get("temperature", 0.0)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "max_tokens": int(cfg.get("max_tokens", 700)),
        "stream": False,
    }

    if cfg.get("stop"):
        body["stop"] = cfg["stop"]

    print(f"[GROQ] tentando modelo: {model_name}")
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = {"text": resp.text[:800]}
            print(f"[GROQ-{model_name}] HTTP {resp.status_code}: {str(err)[:300]}")
            return None
        data = resp.json()
        choices = (data or {}).get("choices") or []
        if not choices or not choices[0].get("message", {}).get("content"):
            return None
        return choices[0]["message"]["content"].strip()
    except Exception as e:
        print(f"[GROQ] erro: {e}")
        return None

# ---------------- Chunking + chamada IA ----------------
H_CHUNK = 8  # chunks pequenos para estabilidade no 120B

def _build_context_message(queixa_atual: str,
                           bloco1_resumo_py: str,
                           historicos_chunk: List[Dict[str, Any]],
                           exams_payload: List[Dict[str, Any]],
                           user_prompt_text: str,
                           part_idx: int,
                           total_parts: int) -> str:
    hist_json = json.dumps(historicos_chunk, ensure_ascii=False)
    exams_json = json.dumps(exams_payload, ensure_ascii=False)
    ctx = (
        f"QUEIXA_ATUAL:\n{queixa_atual}\n\n"
        f"BLOCO1_TEXTO_PY:\n{bloco1_resumo_py}\n\n"
        f"EXAMES_JSON_SANGUE_120D:\n{exams_json}\n\n"
        f"HISTORICO_JSON (PARTE {part_idx}/{total_parts}):\n{hist_json}\n\n"
    )
    # Anexa o prompt do arquivo, sem mudar nada nele.
    prompt = ctx + (user_prompt_text or "")
    # Hard-limit de segurança (aprox 16k chars)
    if len(prompt) > 16000:
        prompt = prompt[-16000:]
    return prompt

def _extract(tag: str, txt: str) -> str:
    m = re.search(rf"<<<{re.escape(tag)}>>>\s*(.*?)\s*<<<FIM_{re.escape(tag)}>>>", txt, re.S|re.I)
    return m.group(1).strip() if m else ""

def _lines(block_txt: str) -> List[str]:
    return [clean_text(x) for x in re.findall(r"(?m)^\-\s+(.*)$", block_txt)]

def call_groq_map_reduce_text(queixa_atual: str,
                              bloco1_resumo_py: str,
                              historicos: List[Dict[str, Any]],
                              model_cfgs: List[Dict[str, Any]],
                              user_prompt_text: str,
                              exams_payload: List[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, Any]]:
    if not model_cfgs:
        return (None, {})
    for cfg in model_cfgs:
        parts = [historicos[i:i+H_CHUNK] for i in range(0, len(historicos), H_CHUNK)] or [[]]
        total_parts = len(parts)
        agg = {"bloco1": [], "bloco2": [], "bloco3": [], "obs": "", "cond": "", "exames_criticos": [], "rodape": []}
        ok_all = True
        for idx, chunk in enumerate(parts, start=1):
            prompt = _build_context_message(queixa_atual, bloco1_resumo_py, chunk, exams_payload, user_prompt_text, idx, total_parts)
            out = groq_chat_text(prompt, cfg)
            if not out:
                ok_all = False
                break
            # Parse por tags do prompt.txt
            b1_txt  = _extract("BLOCO1", out)
            b2_txt  = _extract("BLOCO2", out)
            b3_txt  = _extract("BLOCO3", out)
            b4_obs  = _extract("BLOCO4_OBS", out)
            b4_cond = _extract("BLOCO4_COND", out)
            ex_crit = _extract("EXAMES_CRITICOS", out)
            rodape  = _extract("RODAPE", out)

            agg["bloco1"] += _lines(b1_txt)[:5]
            agg["bloco2"] += _lines(b2_txt)[:5]
            agg["bloco3"] += _lines(b3_txt)
            if b4_obs: agg["obs"] = b4_obs
            if b4_cond: agg["cond"] = b4_cond
            agg["exames_criticos"] += _lines(ex_crit)
            rl = _lines(rodape)
            if rl: agg["rodape"] += rl

        if not ok_all:
            continue

        def _dedup(seq: List[str]) -> List[str]:
            seen, out = set(), []
            for s in [clean_text(x) for x in seq if clean_text(x)]:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out

        agg["bloco1"] = _dedup(agg["bloco1"])[:5]
        agg["bloco2"] = _dedup(agg["bloco2"])[:5]
        agg["bloco3"] = _dedup(agg["bloco3"])[:20]
        agg["rodape"] = _dedup(agg["rodape"])[:30] or ["NENHUM ITEM SEM EVIDÊNCIA"]
        agg["exames_criticos"] = _dedup(agg["exames_criticos"])[:10]
        return (cfg["model"], agg)
    return (None, {})

# ---------------- Fallback Py ----------------
REL_KEYWORDS = [
    "asma","aerolin","salbutamol","salmeterol","formoterol","clenil","beclo","beclometasona",
    "dispneia","falta de ar","sibil","broncoespasmo","broncodilat","pneumo","crise asmática"
]
def find_related_locally(df_hist: pd.DataFrame, queixa_atual: str) -> List[str]:
    if df_hist.empty: return []
    def norm(s): return strip_accents((s or "")).lower()
    qnorm = norm(queixa_atual)
    cols = [c for c in ["queixa","observacao","conduta","especialidade"] if c in df_hist.columns]
    out = []
    for _, r in df_hist.iterrows():
        blob = " ".join([str(r.get(c,"")) for c in cols])
        n = norm(blob)
        if any(k in n for k in REL_KEYWORDS) or any(k in qnorm for k in REL_KEYWORDS):
            dt = r.get("datainicioconsulta")
            dt = pd.to_datetime(dt, errors="coerce")
            dt_s = dt.strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(dt) else ""
            out.append(f"{clean_text(r.get('queixa') or r.get('observacao') or r.get('conduta') or '')} (Posto {r.get('posto')}, id {r.get('idprontuario')}, data {dt_s})")
    seen, uniq = set(), []
    for s in out:
        s2 = clean_text(s)
        if s2 and s2 not in seen:
            seen.add(s2); uniq.append(s2)
    return uniq[:30]

def simple_exams_opinion_py(exams_compact: List[Dict[str, Any]], paciente_idade_anos: Optional[int]) -> str:
    if not exams_compact:
        return "Sem exames de sangue nos últimos 120 dias."
    menor = paciente_idade_anos is not None and paciente_idade_anos < 18
    if menor:
        return f"Foram encontrados {len(exams_compact)} exame(s) de sangue nos últimos 120 dias; faixas de referência de adultos não foram usadas."
    flags = 0
    for r in exams_compact:
        obs = strip_accents(clean_text((r.get("observacao") or "") + " " + (r.get("resultado") or ""))).lower()
        if any(k in obs for k in ["alto","elev","baixo","reduz","reagent","alterad","critico","critica"]):
            flags += 1
    if flags:
        return f"Foram encontrados {len(exams_compact)} exame(s); há {flags} resultado(s) com possível alerta textual."
    return f"Foram encontrados {len(exams_compact)} exame(s) de sangue nos últimos 120 dias, sem alertas textuais óbvios."

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Busca de prontuários + exames de sangue (120d) + análise Groq (texto livre com tags).")
    p.add_argument("-n", "--nome", help="Nome completo do paciente")
    p.add_argument("-d", "--nascimento", help="Data de nascimento dd/mm/yyyy")
    p.add_argument("-q", "--queixa", help="Queixa atual")
    p.add_argument("--desde", help="Buscar prontuários desde (dd/mm/yyyy). Se vazio, 12 meses.")
    p.add_argument("--like", action="store_true", help="Permitir LIKE no nome se igualdade não encontrar.")
    return p.parse_args()

def main():
    ensure_dirs()
    purge_old_files(JSON_DIR, 24)
    purge_old_files(REPORTS_DIR, 24)

    args = parse_args()
    nome = args.nome or input("1) Nome completo do paciente: ").strip()
    data_nasc_str = args.nascimento or input("2) Data de nascimento (dd/mm/yyyy): ").strip()
    queixa = args.queixa or input("3) Queixa atual do paciente: ").strip()
    desde_str = args.desde or input("4) Buscar desde (dd/mm/yyyy): ").strip()

    try:
        nasc_date = datetime.strptime(data_nasc_str, "%d/%m/%Y").date()
    except ValueError:
        print(json.dumps({"ok": False, "erro": "Data de nascimento inválida. Use dd/mm/yyyy."}, ensure_ascii=False))
        return
    idade_anos = int((date.today() - nasc_date).days // 365.25)

    dt_ini_user = None
    if desde_str:
        try:
            dt_ini_user = datetime.strptime(desde_str, "%d/%m/%Y").date()
        except ValueError:
            print(json.dumps({"ok": False, "erro": "Data 'desde' inválida. Use dd/mm/yyyy."}, ensure_ascii=False))
            return
    else:
        dt_ini_user = date.today() - relativedelta(months=12)

    conns = build_conns_from_env()
    if not conns:
        print(json.dumps({"ok": False, "erro": "Nenhum posto configurado no .env."}, ensure_ascii=False))
        return

    # -------- 1) CONSULTAS (define recorte final) --------
    def _run_consultas(dt_inicio: date, like_flag: bool) -> pd.DataFrame:
        frames = []
        for lbl, conn_str in conns.items():
            df = query_posto(lbl, conn_str, nome, nasc_date, dt_inicio, use_like=like_flag)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames.append(df)
        return safe_concat(frames)

    df_all = _run_consultas(dt_ini_user, like_flag=False)
    if df_all.empty and (args.like or True):
        print("Nenhum dado com igualdade exata. Tentando busca parcial (LIKE)...")
        df_all = _run_consultas(dt_ini_user, like_flag=True)

    dt_ini_final = dt_ini_user
    if df_all.empty or len(df_all) < 10:
        print("Poucos registros no recorte. Ampliando busca para 60 meses…")
        dt_ini_5y = date.today().replace(day=1) - relativedelta(months=60)
        df_all_5y = _run_consultas(dt_ini_5y, like_flag=True)
        dt_ini_final = dt_ini_5y
        df_all = df_all_5y

    # Normalização mínima das consultas
    if not df_all.empty:
        df_all.columns = [str(c).strip().lower() for c in df_all.columns]
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente","datainicioconsulta","idprontuario"]:
            if col not in df_all.columns: df_all[col] = None
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente"]:
            df_all[col] = df_all[col].map(clean_text)

    # ------ HISTÓRICO para IA (consultas) ------
    historicos = []
    if not df_all.empty:
        df_hist = df_all.copy()
        def _no_show_row(row: dict) -> bool:
            txt = " ".join([str(row.get(k,"")) for k in ("queixa","observacao","conduta")])
            txt_norm = strip_accents(txt).lower()
            return ("nao compareceu" in txt_norm) and ("chamad" in txt_norm)
        mask_send = []
        for _, row in df_hist.iterrows():
            mask_send.append(not _no_show_row(row.to_dict()))
        df_hist_send = df_hist.loc[mask_send].copy()
        def _is_blank_series(s: pd.Series) -> pd.Series:
            if s is None: return pd.Series([True] * 0)
            return s.isna() | s.astype(str).str.strip().eq("") | s.astype(str).str.lower().eq("none")
        q_blank = _is_blank_series(df_hist_send.get("queixa", pd.Series([None]*len(df_hist_send))))
        c_blank = _is_blank_series(df_hist_send.get("conduta", pd.Series([None]*len(df_hist_send))))
        df_hist_send = df_hist_send.loc[~(q_blank & c_blank)].reset_index(drop=True)
        historicos = sanitize_for_json(df_hist_send).to_dict(orient="records")

    # -------- 2) EXAMES DE SANGUE (SOMENTE 120 DIAS) --------
    dt_ini_exams = _blood_lookback_date()
    df_exams_raw = query_exams_all_posts(conns, nome, nasc_date, dt_ini_exams)
    if not df_exams_raw.empty:
        df_exams_raw.columns = [str(c).strip() for c in df_exams_raw.columns]
    exams_rows = []
    for _, r in (df_exams_raw if not df_exams_raw.empty else pd.DataFrame()).iterrows():
        rdict = {k: r[k] for k in r.index}
        dl = pd.to_datetime(rdict.get("DataLiberado") or rdict.get("dataliberado") or rdict.get("Data") or None, errors="coerce")
        if pd.isna(dl) or dl.date() < dt_ini_exams:
            continue
        if _is_blood_exam_row(rdict):
            exams_rows.append(rdict)
    # Compacto para IA
    def _compact_exam_row(r: Dict[str, Any]) -> Dict[str, Any]:
        keys = {k.lower(): k for k in r.keys()}
        def g(*opts):
            for o in opts:
                if o.lower() in keys:
                    return r[keys[o.lower()]]
            return None
        def trunc(s, n=120):
            s = clean_text(s)
            return (s[:n] + "…") if s and len(s) > n else s
        return {
            "posto": r.get("posto"),
            "dataliberado": g("dataliberado","dataresultado","data","datacoleta","datalancamento"),
            "grupo": g("grupo"),
            "servico": g("servico"),
            "exame": g("exame","exameitem","examedescricao"),
            "resultado": trunc(g("exameresultado","resultado")),
            "referencia": trunc(g("referencia","valorreferencia","intervaloreferencia")),
            "unidade": g("unidade","unidademedida"),
            "observacao": trunc(g("exameobservacao","observacao","observacoes")),
            "idlancamentoservico": g("idlancamentoservico"),
            "idresultado": g("idresultado"),
            "item": g("item"),
        }
    exams_compact_for_prompt: List[Dict[str, Any]] = [_compact_exam_row(r) for r in exams_rows[:120]]

    # -------- 3) Bloco 1 (fallback Py) --------
    bloco1_list_py = build_block1_list_py(df_all)  # lista organizada para exibir
    if not df_all.empty:
        df_b1 = build_last10(df_all)
        b1_lines = df_b1["linha_fmt"].tolist()
    else:
        b1_lines = []
    bloco1_resumo_py = summarize_block1_lines(b1_lines, maxlen=1000)

    # -------- 4) Auditoria JSON bruto (salvo em arquivo, não no output) --------
    payload_full = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "data_inicial_prontuarios": dt_ini_final.isoformat(),
            "exames_sangue_lookback_dias": EXAMS_BLOOD_LOOKBACK_DAYS,
            "gerado_em": datetime.now().isoformat(),
            "registros_total_consultas": int(len(df_all)) if not isinstance(df_all, list) else 0,
            "registros_total_exames_sangue_120d": int(len(exams_rows)),
        },
        "consultas_para_ia": historicos,
        "exames_sangue_120d_compacto_para_ia": exams_compact_for_prompt,
    }
    json_path = save_json(payload_full)
    print(f"JSON: {os.path.basename(json_path)} (será limpo em ~1h)")

    # -------- 5) IA (GROQ, texto livre com tags do prompt.txt) --------
    prompt_txt = load_user_prompt()
    model_cfgs = _list_groq_model_configs()
    model_used, result_ia = call_groq_map_reduce_text(
        queixa, bloco1_resumo_py, historicos, model_cfgs, prompt_txt, exams_compact_for_prompt
    )

    # -------- 6) Montagem dos blocos ----------
    titulo_b1 = "BLOCO1 - RESUMO DOS ÚLTIMOS ATENDIMENTOS (IA)" if (result_ia and result_ia.get("bloco1")) else "BLOCO1 - RESUMO DOS ÚLTIMOS ATENDIMENTOS (Py)"
    titulo_b2 = "BLOCO2 - ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL (IA)" if (result_ia and result_ia.get("bloco2")) else "BLOCO2 - ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL (Py)"
    titulo_b3 = "BLOCO3 - DIAGNOSTICO DIFERENCIAL (IA)" if (result_ia and result_ia.get("bloco3")) else "BLOCO3 - DIAGNOSTICO DIFERENCIAL (Py)"
    titulo_b4 = "BLOCO4 - SUGESTAO CAMPOS OBS E CONDUTA (IA)" if (result_ia and (result_ia.get("obs") or result_ia.get("cond"))) else "BLOCO4 - SUGESTAO CAMPOS OBS E CONDUTA (Py)"
    titulo_rod = "RODAPE (IA)" if (result_ia and result_ia.get("rodape")) else "RODAPE (Py)"

    # BLOCO1 conteúdo final (lista IA -> senão lista Py)
    if result_ia and result_ia.get("bloco1"):
        bloco1_list_final = result_ia["bloco1"][:5]
    else:
        bloco1_list_final = bloco1_list_py

    # BLOCO2/3/4/rodape
    bloco2 = result_ia.get("bloco2", []) if result_ia else []
    bloco3 = result_ia.get("bloco3", []) if result_ia else []
    obs_ia  = result_ia.get("obs", "") if result_ia else ""
    cond_ia = result_ia.get("cond", "") if result_ia else ""
    rodape  = result_ia.get("rodape", ["NENHUM ITEM SEM EVIDÊNCIA"]) if result_ia else ["NENHUM ITEM SEM EVIDÊNCIA"]
    exames_criticos = result_ia.get("exames_criticos", []) if result_ia else []

    # Fallbacks Py se IA não entregou
    if not bloco2:
        bloco2 = find_related_locally(df_all, queixa)
        titulo_b2 = "BLOCO2 - ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL (Py)"
    bloco4 = {"observacao": obs_ia, "conduta": cond_ia}
    if not bloco4["observacao"]:
        parecer_py = simple_exams_opinion_py(exams_compact_for_prompt, idade_anos)
        bloco4["observacao"] = parecer_py

    # -------- 7) Saída FINAL (sem listar exames, exceto se críticos) --------
    def fmt_ddmmyyyy(d: date) -> str:
        try:
            return datetime.strptime(str(d), "%Y-%m-%d").strftime("%d%m%Y")
        except Exception:
            try:
                return d.strftime("%d%m%Y")
            except Exception:
                return str(d)

    output = {
        "paciente": nome,
        "data_nascimento": nasc_date.isoformat(),
        "queixa_atual": queixa,
        "prontuarios_total": int(len(df_all)) if not df_all.empty else 0,
        "Possui exames": bool(exams_compact_for_prompt),
        "data_inicial": fmt_ddmmyyyy(dt_ini_final),

        "BLOCO1 - ": titulo_b1,
        "RESUMO DOS ÚLTIMOS ATENDIMENTOS": bloco1_list_final,

        "BLOCO2 - ": titulo_b2,
        "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": bloco2,

        "BLOCO3 - ": titulo_b3,
        "DIAGNOSTICO DIFERENCIAL": bloco3,

        "BLOCO4 - ": titulo_b4,
        "SUGESTAO CAMPOS OBS E CONDUTA": bloco4,

        "RODAPE - ": titulo_rod,
        "rodape": rodape,

        "provider_mode": "ia_text" if model_used else "py_fallback",
        "provider_used": model_used,
    }
    if exames_criticos:
        output["EXAMES MUITO IMPORTANTES (IA)"] = exames_criticos

    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
