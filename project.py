# -*- coding: utf-8 -*-
"""
project.py — Busca prontuários multi-postos + exames (120d) e análise Groq (texto).
- BLOCO 1 e BLOCO 2 SEMPRE normatizados, top-5 mais recentes, formato fixo.
- Sem gatilhos respiratórios; relacionamento por similaridade Jaccard.
- Groq 120B: remove campos não suportados; diagnóstico claro; fallback para 70B.

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

# ---------------- UX / Progress ----------------
def _ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

GROQ_DEBUG        = _env("GROQ_DEBUG", "0") in {"1","true","yes","y"}
GROQ_ONLY_120B    = _env("GROQ_ONLY_120B", "0") in {"1","true","yes","y"}
GROQ_TIMEOUT      = int(_env("GROQ_TIMEOUT", "120"))
GROQ_FORCE_TEXT   = _env("GROQ_FORCE_TEXT", "1") in {"1","true","yes","y"}
GROQ_HCHUNK       = max(3, int(_env("GROQ_HCHUNK", "10")))  # reduz partes para 120B não estourar

def dbg(msg: str):
    if GROQ_DEBUG:
        print(f"[DBG {_ts()}] {msg}")

class TermUX:
    def __init__(self):
        self._w = 28  # largura mini-barra
        self._log_path = None

    def attach_log(self, path: str):
        self._log_path = path
        self.log(f"LOG iniciado em {path}")

    def log(self, line: str):
        try:
            if self._log_path:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{_ts()}] {line}\n")
        except Exception:
            pass

    def _bar(self, pct):
        done = int(self._w * pct)
        return f"[{'#'*done}{'.'*(self._w-done)}]{pct*100:5.1f}%"

    def step(self, title: str, i: int, n: int):
        print(f"\n== {i}/{n} {title}")
        self.log(f"STEP {i}/{n}: {title}")

    def tick(self, msg: str, k: int, total: int):
        pct = (k/total) if total else 1.0
        bar = self._bar(pct)
        print(f"  {bar}  {msg}")
        self.log(msg)

UX = TermUX()

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
        print(f"[{lbl}-EX] Conectando...")
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

# ---------------- BLOCO 1 (normatizado) ----------------
def _mk_resumo_min(r: dict) -> str:
    for k in ("queixa","observacao","conduta"):
        v = clean_text(r.get(k))
        if v:
            return v.rstrip(".") + "."
    return ""

def _valid_row_b1(r: dict) -> bool:
    try:
        if not r.get("idprontuario"): return False
        if int(str(r.get("idprontuario")).strip()) <= 0: return False
    except:
        return False
    dt = pd.to_datetime(r.get("datainicioconsulta"), errors="coerce")
    if pd.isna(dt) or dt.to_pydatetime().date() > date.today():
        return False
    blob = " ".join([clean_text(r.get(k)) for k in ("queixa","observacao","conduta")])
    b = strip_accents(blob).lower()
    if b and re.fullmatch(r"(teste\.?\s*){1,}", b):
        return False
    return True

def build_block1_normalizado(df_all: pd.DataFrame, k=5) -> List[str]:
    if df_all is None or df_all.empty:
        return []
    for c in ["posto","idprontuario","datainicioconsulta","queixa","observacao","conduta"]:
        if c not in df_all.columns:
            df_all[c] = None
    df = df_all.copy()
    df["_dt"] = pd.to_datetime(df["datainicioconsulta"], errors="coerce")
    df = df[df.apply(_valid_row_b1, axis=1)].copy()
    if df.empty:
        return []
    df = df.sort_values("_dt", ascending=False).head(k)
    out = []
    for _, r in df.iterrows():
        dt_iso = pd.to_datetime(r["datainicioconsulta"], errors="coerce")
        dt_iso = dt_iso.strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(dt_iso) else ""
        resumo = _mk_resumo_min(r.to_dict())
        posto = clean_text(r.get("posto"))
        rid   = str(r.get("idprontuario")).strip()
        out.append(f"(Posto {posto}, id {rid}, data {dt_iso}) - {resumo}")
    return out

# ---------------- Regras de EXAMES DE SANGUE ----------------
def _is_blood_exam_row(r: Dict[str, Any]) -> bool:
    txt = " ".join([
        str(r.get(k,"")) for k in ["material","grupo","setor","servico","exame","item","servicoexamematerial"]
        if k in r
    ])
    t = strip_accents(clean_text(txt)).lower()
    if "sang" in t:
        return True
    keys = ["hemograma","hemat","bioquim","hormon","serol","imuno","coagul","ferro","glic","perfil lipid","lipid"]
    return any(k in t for k in keys)

def _blood_lookback_date() -> date:
    return (date.today() - timedelta(days=EXAMS_BLOOD_LOOKBACK_DAYS))

# ---------------- Bloco 1 (fallback Py texto livre curto) ----------------
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

# ---------------- Prompt loader ----------------
def load_user_prompt() -> str:
    return _read_file_any_encoding(PROMPT_FILE).strip() if os.path.isfile(PROMPT_FILE) else ""

# ---------------- GROQ models (.ini) ----------------
def _list_groq_model_configs() -> List[Dict[str, Any]]:
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
                "json_mode": False,
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
        items_ordered = [
            {"order": 1, "model": "openai/gpt-oss-120b", "temperature": 0.0, "top_p": 1.0, "max_tokens": 700, "json_mode": False, "stop": None, "name": "default-120b"},
            {"order": 2, "model": "llama-3.3-70b-versatile", "temperature": 0.2, "top_p": 0.9, "max_tokens": 900, "json_mode": False, "stop": None, "name": "fallback-70b"},
        ]
    if GROQ_ONLY_120B:
        items_ordered = [cfg for cfg in items_ordered if cfg["model"] == "openai/gpt-oss-120b"]
    return items_ordered

# ---------------- Persistência do prompt enviado ----------------
def write_sent_files(prompt_text: str, model: str, part_idx: int, total_parts: int) -> str:
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sent_main = os.path.join(REPORTS_DIR, f"sended_{ts}.txt")
    with open(sent_main, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    sent_part = os.path.join(REPORTS_DIR, f"sent_{ts}_{model.replace('/','_')}_p{part_idx}of{total_parts}.txt")
    with open(sent_part, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    return sent_part

# ---------------- Groq HTTP client (TEXTO) ----------------
def groq_chat_text(prompt: str, cfg: Dict[str, Any], part_idx: int, total_parts: int) -> Optional[str]:
    api_key = _env("GROQ_API_KEY", "")
    if not api_key:
        print("[GROQ] falta GROQ_API_KEY.")
        return None

    model_name = cfg["model"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Persistir prompt enviado
    sent_file_path = write_sent_files(prompt, model_name, part_idx, total_parts)
    dbg(f"Prompt salvo em: {sent_file_path}")

    def _do_request(max_tokens: int, attempt: int) -> Tuple[Optional[str], Optional[str], dict]:
        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Você é um assistente clínico objetivo. Responda em pt-BR."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(cfg.get("temperature", 0.0)),
            "top_p": float(cfg.get("top_p", 1.0)),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        # 120B: NÃO enviar campos não suportados; opcionalmente forçar texto
        if model_name == "openai/gpt-oss-120b" and _env("GROQ_FORCE_TEXT","1") in {"1","true","yes","y"}:
            body["response_format"] = {"type": "text"}
        if cfg.get("stop"):
            body["stop"] = cfg["stop"]

        print(f"[GROQ] tentando modelo: {model_name} (attempt {attempt}, max_tokens={body['max_tokens']})")
        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=int(_env("GROQ_TIMEOUT","120")))
            dt_ms = int((time.time() - t0) * 1000)
            dbg(f"[GROQ HTTP] status={resp.status_code} in {dt_ms}ms | model={model_name} | attempt={attempt}")

            if resp.status_code != 200:
                try: err_json = resp.json()
                except Exception: err_json = {"text": resp.text[:2000]}
                print(f"[GROQ-{model_name}] HTTP {resp.status_code}: {(str(err_json)[:300])}")
                with open(os.path.join(REPORTS_DIR, f"groq_error_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name.replace('/','_')}_p{part_idx}_a{attempt}.txt"), "w", encoding="utf-8") as f:
                    f.write(json.dumps(err_json, ensure_ascii=False, indent=2) if isinstance(err_json, dict) else str(err_json))
                return None, None, {}
            # Guardar a resposta crua
            raw_path = os.path.join(REPORTS_DIR, f"groq_raw_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name.replace('/','_')}_p{part_idx}_a{attempt}.json")
            try:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
            except Exception:
                pass

            data = resp.json()
            choices = (data or {}).get("choices") or []
            msg = choices[0].get("message", {}) if choices else {}
            content = (msg.get("content") or "").strip()
            finish_reason = choices[0].get("finish_reason")

            if content:
                with open(os.path.join(REPORTS_DIR, f"groq_reply_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name.replace('/','_')}_p{part_idx}_a{attempt}.txt"), "w", encoding="utf-8") as f:
                    f.write(content)
            return content if content else None, str(finish_reason) if finish_reason else None, data
        except requests.Timeout:
            print(f"[GROQ-{model_name}] erro: timeout ({int((time.time()-t0)*1000)}ms).")
            return None, None, {}
        except Exception as e:
            print(f"[GROQ-{model_name}] erro: {e}")
            with open(os.path.join(REPORTS_DIR, f"groq_exception_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name.replace('/','_')}_p{part_idx}_a{attempt}.txt"), "w", encoding="utf-8") as f:
                f.write(repr(e))
            return None, None, {}

    # 1ª tentativa com o max_tokens do .ini
    base_max = int(cfg.get("max_tokens", 700))
    content, finish_reason, data = _do_request(base_max, attempt=1)
    if content:
        return content

    # Se 120B cortou por length e veio vazio, faz retry “alavancado”
    if model_name == "openai/gpt-oss-120b" and (finish_reason or "").lower() == "length":
        # sobe teto de saída mas mantendo compatibilidade (limita em 2048)
        boosted = min(2048, int(float(_env("GROQ_RETRY_MAXTOKENS","1500"))))
        print(f"[GROQ-{model_name}] resposta sem content; finish_reason=length. Retentando com max_tokens={boosted}.")
        content2, finish_reason2, data2 = _do_request(boosted, attempt=2)
        if content2:
            return content2
        # log de diagnóstico agregado
        diag = {
            "first_attempt_finish_reason": finish_reason,
            "second_attempt_finish_reason": finish_reason2,
            "advice": "120B ainda cortou a saída. Fallback para modelo secundário (70B)."
        }
        with open(os.path.join(REPORTS_DIR, f"groq_diag_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name.replace('/','_')}_p{part_idx}.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(diag, ensure_ascii=False, indent=2))
        return None

    # Outros casos: retorna None e o chamador decide fallback
    if (finish_reason or "").lower() == "length":
        print(f"[GROQ-{model_name}] finish_reason=length e sem conteúdo. Considerar reduzir chunk/prompt.")
    return None

# ---------------- Chunking + chamada IA (texto livre) ----------------
H_CHUNK = GROQ_HCHUNK  # <= mais conservador para 120B

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
    return ctx + (user_prompt_text or "")

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
    print("\n== 5/5 Analisando com IA (Groq)")
    dbg(f"Históricos={len(historicos)} | chunk={H_CHUNK}")
    for cfg in model_cfgs:
        parts = [historicos[i:i+H_CHUNK] for i in range(0, len(historicos), H_CHUNK)] or [[]]
        total_parts = len(parts)
        agg = {"bloco1": [], "bloco2": [], "bloco3": [], "obs": "", "cond": "", "exames_criticos": [], "rodape": []}
        ok_all = True
        for idx, chunk in enumerate(parts, start=1):
            prompt = _build_context_message(queixa_atual, bloco1_resumo_py, chunk, exams_payload, user_prompt_text, idx, total_parts)
            out = groq_chat_text(prompt, cfg, idx, total_parts)
            if not out:
                ok_all = False
                break
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
            if b4_obs:  agg["obs"]  = b4_obs
            if b4_cond: agg["cond"] = b4_cond
            agg["exames_criticos"] += _lines(ex_crit)
            rl = _lines(rodape)
            if rl: agg["rodape"] += rl

            print(f"  [{'#'*int(28*(idx/total_parts))}{'.'*int(28*(1-idx/total_parts))}]{(idx/total_parts)*100:5.1f}%  Parte {idx}/{total_parts} processada")
            dbg(f"OUT head: {(out[:220] if out else '')!r}")

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
        agg["bloco3"] = _dedup(agg["bloco3"])[:30]
        agg["rodape"] = _dedup(agg["rodape"])[:30] or ["NENHUM ITEM SEM EVIDÊNCIA"]
        agg["exames_criticos"] = _dedup(agg["exames_criticos"])[:10]
        return (cfg["model"], agg)
    return (None, {})

# ---------------- Similaridade (Jaccard) p/ BLOCO 2 ----------------
PT_STOPWORDS = {
    "a","o","as","os","um","uma","de","do","da","dos","das","para","por","em","no","na","nos","nas",
    "e","ou","com","sem","ao","aos","à","às","que","qual","quais","se","sua","seu","suas","seus",
    "há","ha","tem","têm","ter","foi","era","ser","está","esta","estar","são","sao","já","ja","não","nao",
    "hoje","ontem","amanha","amanhã","dias","dia","semana","mes","mês","meses","ano","anos","deu","teve"
}

def _tokens_pt(s: str) -> set:
    s = strip_accents(clean_text(s)).lower()
    toks = re.findall(r"[a-z0-9]+", s)
    return {t for t in toks if t not in PT_STOPWORDS and len(t) > 2}

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    u = a | b
    if not u: return 0.0
    return len(a & b) / len(u)

def build_block2_related(df_hist: pd.DataFrame, queixa_atual: str, k=5, thr=0.6) -> List[str]:
    if df_hist is None or df_hist.empty:
        return []
    for c in ["posto","idprontuario","datainicioconsulta","queixa","observacao","conduta"]:
        if c not in df_hist.columns:
            df_hist[c] = None
    tq = _tokens_pt(queixa_atual)
    rows = []
    for _, r in df_hist.iterrows():
        txt = " ".join([str(r.get(c,"")) for c in ("queixa","observacao","conduta")])
        tr = _tokens_pt(txt)
        score = _jaccard(tq, tr)
        if score >= thr:
            rows.append((pd.to_datetime(r.get("datainicioconsulta"), errors="coerce"), r, score))
    if not rows:
        return []
    rows = [(dt, r, sc) for (dt, r, sc) in rows if pd.notna(dt)]
    rows.sort(key=lambda x: (x[0], x[2]), reverse=True)  # data desc, depois score
    out = []
    for dt, r, _ in rows[:k]:
        dt_iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
        resumo = _mk_resumo_min(r.to_dict())
        posto = clean_text(r.get("posto"))
        rid   = str(r.get("idprontuario")).strip()
        out.append(f"(Posto {posto}, id {rid}, data {dt_iso}) - {resumo}")
    return out

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
    p = argparse.ArgumentParser(description="Busca de prontuários + exames (120d) + análise Groq (texto com tags).")
    p.add_argument("-n", "--nome", help="Nome completo do paciente")
    p.add_argument("-d", "--nascimento", help="Data de nascimento dd/mm/yyyy")
    p.add_argument("-q", "--queixa", help="Queixa atual")
    p.add_argument("--desde", help="Buscar prontuários desde (dd/mm/yyyy). Se vazio, 12 meses.")
    p.add_argument("--like", action="store_true", help="Permitir LIKE no nome se igualdade não encontrar.")
    p.add_argument("--jaccard", type=float, default=float(_env("B2_JACCARD_THR","0.6")), help="Limite Jaccard para BLOCO 2 (0-1).")
    return p.parse_args()

def main():
    ensure_dirs()
    purge_old_files(JSON_DIR, 24)
    purge_old_files(REPORTS_DIR, 24)

    groq_log_path = os.path.join(REPORTS_DIR, f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    UX.attach_log(groq_log_path)
    dbg(f"Groq log file: {groq_log_path}")

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

    # -------- 1) CONSULTAS --------
    UX.step("Consultando prontuários nos postos", 1, 5)
    def _run_consultas(dt_inicio: date, like_flag: bool) -> pd.DataFrame:
        frames = []
        total = len(conns)
        for idx, (lbl, conn_str) in enumerate(conns.items(), start=1):
            UX.tick(f"[{lbl}] conectando…", idx-1, total)
            df = query_posto(lbl, conn_str, nome, nasc_date, dt_inicio, use_like=like_flag)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames.append(df)
                UX.tick(f"[{lbl}] ✓ {len(df)} registro(s)", idx, total)
                UX.log(f"[{lbl}] colunas={list(df.columns)}")
            else:
                UX.tick(f"[{lbl}] ✓ nenhum registro", idx, total)
        return safe_concat(frames)

    df_all = _run_consultas(dt_ini_user, like_flag=False)
    if df_all.empty and (args.like or True):
        print("Nenhum dado com igualdade exata. Tentando busca parcial (LIKE)...")
        df_all = _run_consultas(dt_ini_user, like_flag=True)

    dt_ini_final = dt_ini_user
    if df_all.empty or len(df_all) < 10:
        print("")
        UX.step("Poucos registros — ampliando recorte para 60 meses", 2, 5)
        dt_ini_5y = date.today().replace(day=1) - relativedelta(months=60)
        df_all_5y = _run_consultas(dt_ini_5y, like_flag=True)
        dt_ini_final = dt_ini_5y
        df_all = df_all_5y

    if not df_all.empty:
        df_all.columns = [str(c).strip().lower() for c in df_all.columns]
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente","datainicioconsulta","idprontuario","posto"]:
            if col not in df_all.columns: df_all[col] = None
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente","posto"]:
            df_all[col] = df_all[col].map(clean_text)

    # ------ HISTÓRICO para IA ------
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

    # -------- 2) EXAMES (120d) --------
    UX.step("Coletando exames de sangue (120 dias)", 3, 5)
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

    exams_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exams_json_path = os.path.join(REPORTS_DIR, f"exams_{exams_ts}.json")
    with open(exams_json_path, "w", encoding="utf-8") as f:
        json.dump(exams_compact_for_prompt, f, ensure_ascii=False, indent=2)
    print(f"[EXAMES] JSON gerado: {os.path.basename(exams_json_path)}")

    # -------- 3) Bloco 1 (texto Py para contexto da IA) --------
    if not df_all.empty:
        df_b1 = build_last10(df_all)
        b1_lines = df_b1["linha_fmt"].tolist()
    else:
        b1_lines = []
    bloco1_resumo_py_txt = summarize_block1_lines(b1_lines, maxlen=1000)

    # -------- 4) Auditoria JSON bruto --------
    UX.step("Gerando JSON de auditoria", 4, 5)
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
    UX.tick(f"Arquivo salvo: {os.path.basename(json_path)} (limpeza ~24h)", 1, 1)

    # -------- 5) IA (Groq, TEXTO) --------
    prompt_txt = load_user_prompt()
    model_cfgs = _list_groq_model_configs()
    model_used, result_ia = call_groq_map_reduce_text(
        queixa, bloco1_resumo_py_txt, historicos, model_cfgs, prompt_txt, exams_compact_for_prompt
    )

    # -------- 6) Montagem dos blocos ----------
    # Títulos transparentes
    usou_ia = bool(model_used and result_ia)
    titulo_b1 = "BLOCO1 - RESUMO DOS ÚLTIMOS ATENDIMENTOS (IA)" if usou_ia else "BLOCO1 - RESUMO DOS ÚLTIMOS ATENDIMENTOS (Py)"
    titulo_b3 = "DIAGNOSTICO DIFERENCIAL (IA)" if (result_ia and result_ia.get("bloco3")) else "DIAGNOSTICO DIFERENCIAL (Py)"
    titulo_b4 = "SUGESTAO CAMPOS OBS E CONDUTA (IA)" if (result_ia and (result_ia.get("obs") or result_ia.get("cond"))) else "SUGESTAO CAMPOS OBS E CONDUTA (Py)"
    titulo_rd = "RODAPE (IA)" if (result_ia and result_ia.get("rodape")) else "RODAPE (Py)"

    # BLOCO 1: SEMPRE normatizado do dado (top-5 mais recentes)
    b1_lines_norm = build_block1_normalizado(df_all, k=5)
    bloco1_final = "\n".join(b1_lines_norm) if b1_lines_norm else bloco1_resumo_py_txt

    # BLOCO 2: SEMPRE normatizado por Jaccard (sem gatilhos), top-5 mais recentes
    bloco2 = build_block2_related(df_all, queixa, k=5, thr=args.jaccard)
    titulo_b2 = "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL (Py)"

    # BLOCO3/4/rodapé da IA (se houver)
    bloco3 = (result_ia.get("bloco3", []) if result_ia else [])
    obs_ia  = (result_ia.get("obs", "") if result_ia else "")
    cond_ia = (result_ia.get("cond", "") if result_ia else "")
    rodape  = (result_ia.get("rodape", ["NENHUM ITEM SEM EVIDÊNCIA"]) if result_ia else ["NENHUM ITEM SEM EVIDÊNCIA"])
    exames_criticos = (result_ia.get("exames_criticos", []) if result_ia else [])

    bloco4 = {"observacao": obs_ia, "conduta": cond_ia}
    if not bloco4["observacao"]:
        bloco4["observacao"] = simple_exams_opinion_py(exams_compact_for_prompt, idade_anos)

    # -------- 7) Saída FINAL --------
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
        "RESUMO DOS ÚLTIMOS ATENDIMENTOS": bloco1_final,

        "BLOCO2 - ": titulo_b2,
        "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": bloco2,

        "BLOCO3 - ": titulo_b3,
        "DIAGNOSTICO DIFERENCIAL": bloco3,

        "BLOCO4 - ": titulo_b4,
        "SUGESTAO CAMPOS OBS E CONDUTA": bloco4,

        "RODAPE - ": titulo_rd,
        "rodape": rodape,
        "provider_mode": "ia_text" if model_used else "py_fallback",
        "provider_used": model_used,
    }
    if exames_criticos:
        output["EXAMES MUITO IMPORTANTES (IA)"] = exames_criticos

    print("\n== RESULTADO ==")
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
