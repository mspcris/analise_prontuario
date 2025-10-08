# -*- coding: utf-8 -*-
"""
project.py — Busca prontuários multi-postos e exames (com mesmo recorte temporal),
resume localmente os últimos atendimentos (Bloco 1, Py) e gera análise via Groq (Blocos 2–4, IA, com fallback Py).

Mudanças pedidas:
- A janela de 60 meses NÃO é só para o Bloco 1: se houver <10 consultas no recorte informado,
  refaz TUDO (todas as consultas) com 60 meses e usa ESSA mesma data para BUSCAR EXAMES também.
- O SELECT de EXAMES só roda DEPOIS de finalizar o recorte de consultas (para herdar o dt_ini_final).
- Bloco 1 (Py) com título “RESUMO DOS ÚLTIMOS ATENDIMENTOS (Py)”, formato por linha:
  “Em dd/mm/aaaa, na {especialidade}, registrou-se: …”; alvo 500 chars, limite rígido 1000 chars.
- “ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL”: se IA vier vazio, há fallback (Py) por palavras-chave (asma, aerolin, salbutamol, dispneia, etc.) com fontes.
- Ao lado de cada título, informar (Py) ou (IA). Saída inclui chaves de título para clareza, mantendo chaves antigas para compatibilidade.
- Antes de salvar CSV de exames (opcional), montar JSON de exames e enviá-lo à IA (sempre respeitando dt_ini_final).
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
    # Você disse que mudou o select do laboratório; o arquivo será lido daqui.
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

def query_exams_all_posts(conns: Dict[str,str], paciente: str, nasc_date: date, dt_ini: date) -> pd.DataFrame:
    sql_txt = load_exam_sql()
    frames = []
    for lbl, conn_str in conns.items():
        try:
            engine = make_engine(conn_str)
            with engine.begin() as con:
                df = pd.read_sql(text(sql_txt), con=con, params={"paciente": paciente.strip(), "nasc": nasc_date, "dt_ini": dt_ini})
            if not df.empty:
                df.insert(0, "posto", lbl)
                print(f"[{lbl}-EX] {len(df)} resultado(s)")
                frames.append(df)
            else:
                print(f"[{lbl}-EX] Nenhum resultado")
        except Exception as e:
            msg = str(e)
            if "916" in msg or "no contexto de segurança atual" in msg or "contexto de segurança atual" in msg:
                print(f"[{lbl}-EX] Sem permissão para ler exames (916). Ignorado.")
            else:
                print(f"[{lbl}-EX] ERRO: {msg.splitlines()[0]}")
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

# ---------------- Bloco 1 (Py) ----------------
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
    df["linha_fmt"] = df.apply(lambda r: f"Em {r['data_fmt']}, na {clean_text(r.get('especialidade') or '').lower()}, registrou-se: { _mk_resumo(r) }".strip(), axis=1)
    return df

def summarize_block1_lines(lines: List[str], target=500, maxlen=1000) -> str:
    # Junta linhas mantendo cortes; corta duro em 1000.
    txt = "RESUMO DOS ÚLTIMOS ATENDIMENTOS (Py)\n\n" + "\n".join([l for l in lines if l])
    if len(txt) <= maxlen:
        return txt
    # estratégia simples: cortar no limite preservando linhas inteiras
    head = "RESUMO DOS ÚLTIMOS ATENDIMENTOS (Py)\n\n"
    budget = maxlen - len(head)
    out_lines, acc = [], 0
    for l in lines:
        L = len(l) + 1  # +\n
        if acc + L > budget:
            break
        out_lines.append(l); acc += L
    if not out_lines:
        # como fallback, trunca a primeira linha
        return (head + (lines[0][:budget])).rstrip()
    return (head + "\n".join(out_lines)).rstrip()

# ---------------- Prompt ----------------
def load_user_prompt() -> str:
    if not os.path.isfile(PROMPT_FILE):
        return (
            "# STRICT_EVIDENCE_PROMPT (fallback)\n"
            "Responda SOMENTE JSON com chaves: bloco2 (lista), bloco3 (lista), "
            "bloco4 {observacao,conduta}, rodape (lista). Quando citar algo do histórico, inclua fonte.\n"
        )
    return _read_file_any_encoding(PROMPT_FILE).strip()

def _compact_exam_row(r: Dict[str, Any]) -> Dict[str, Any]:
    keys = {k.lower(): k for k in r.keys()}
    def g(*opts):
        for o in opts:
            if o.lower() in keys:
                return r[keys[o.lower()]]
        return None
    def trunc(s, n=300):
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

def build_prompt_chunk_pyblock(queixa_atual: str,
                               bloco1_resumo_py: str,
                               historicos_chunk: List[Dict[str, Any]],
                               user_prompt: str,
                               part_idx: int,
                               part_total: int,
                               exams_payload: List[Dict[str, Any]]) -> str:
    hist_json = json.dumps(historicos_chunk, ensure_ascii=False)
    exams_json = json.dumps(exams_payload, ensure_ascii=False)
    header = (
        f"QUEIXA_ATUAL:\n{queixa_atual}\n\n"
        f"BLOCO1_RESUMO_PY (NÃO ALTERAR):\n{bloco1_resumo_py}\n\n"
        f"EXAMES_JSON (use quando relevante):\n{exams_json}\n\n"
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
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 3000,
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
                def getf(key, default=None): return g.get(key, fallback=default)
                model = (getf("model","") or "").strip().strip('"').strip("'")
                if model: cfg["model"] = model
                try: cfg["temperature"] = float(getf("temperature", cfg["temperature"]))
                except: pass
                try: cfg["top_p"] = float(getf("top_p", cfg["top_p"]))
                except: pass
                try: cfg["max_tokens"] = int(getf("max_tokens", cfg["max_tokens"]))
                except: pass
                eff = (getf("reasoning_effort","") or "").strip().strip('"').strip("'")
                cfg["reasoning_effort"] = eff
                jm = (str(getf("json_mode","true"))).lower()
                cfg["json_mode"] = jm in {"1","true","yes","y"}
                stop_raw = getf("stop","")
                if stop_raw:
                    try: cfg["stop"] = json.loads(stop_raw)
                    except: cfg["stop"] = [s.strip() for s in re.split(r"[;,]\s*", stop_raw.strip(" []")) if s.strip()]
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
            try: body = e.response.json()
            except Exception: body = e.response.text if e.response is not None else ""
            print(f"[{label}] HTTP {getattr(e.response,'status_code',None)}: {str(body)[:300]}")
            retry_after = 0.0
            if e.response is not None:
                ra = e.response.headers.get("Retry-After")
                try: retry_after = float(ra)
                except: retry_after = 0.0
            if i < attempts - 1:
                delay = max(retry_after, (backoff ** i)) + random.uniform(0,0.4)
                print(f"[{label}] retry em {delay:.1f}s...")
                time.sleep(delay); continue
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
    max_toks = int(cfg.get("max_tokens", 3000))
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Você é um assistente clínico objetivo. Responda em pt-BR. Não invente."},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.get("temperature", 0.0),
        "top_p": cfg.get("top_p", 1),
        "max_tokens": max_toks,
        "stream": False,
    }
    if cfg.get("json_mode", True):
        body["response_format"] = {"type": "json_object"}
    if cfg.get("stop"):
        body["stop"] = cfg["stop"]
    eff = (cfg.get("reasoning_effort") or "").strip()
    if eff: body["reasoning"] = {"effort": eff}

    label = f"GROQ-{model_name}"
    print(f"[GROQ] tentando modelo: {model_name}")

    def _do():
        resp = requests.post(url, headers=headers, json=body, timeout=90)
        if resp.status_code != 200:
            try: print(f"[{label}] HTTP {resp.status_code}: {resp.json()}")
            except Exception: print(f"[{label}] HTTP {resp.status_code}: {resp.text[:800]}")
            resp.raise_for_status()
        data = resp.json()
        choices = (data or {}).get("choices") or []
        if not choices or not choices[0].get("message", {}).get("content"):
            raise RuntimeError("groq_empty_or_malformed")
        return choices[0]["message"]["content"].strip()

    return _request_with_retries(_do, label=label, attempts=3, backoff=1.5)

def _safe_json_loads(txt: str) -> Optional[dict]:
    if not txt: return None
    s = txt.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).rstrip("`").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try: return json.loads(s[i:j+1])
        except Exception: return None
    return None

# ---------------- Map → Reduce (chunking) ----------------
def call_groq_map_reduce(queixa_atual: str,
                         bloco1_resumo_py: str,
                         historicos: List[Dict[str, Any]],
                         model_cfgs: List[Dict[str, Any]],
                         user_prompt: str,
                         exams_payload: List[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, Any]]:
    if not model_cfgs:
        return (None, {})
    for cfg in model_cfgs:
        CHUNK_MAX_ITEMS = 120
        parts = [historicos[i:i+CHUNK_MAX_ITEMS] for i in range(0, len(historicos), CHUNK_MAX_ITEMS)]
        total_parts = max(1, len(parts))
        agg = {"bloco2": [], "bloco3": [], "bloco4": {"observacao":"","conduta":""}, "rodape": []}
        ok_all = True
        for idx, chunk in enumerate(parts, start=1):
            prompt = build_prompt_chunk_pyblock(queixa_atual, bloco1_resumo_py, chunk, user_prompt, idx, total_parts, exams_payload)
            out = groq_chat_json(prompt, cfg)
            if not out: ok_all = False; break
            parsed = _safe_json_loads(out)
            if not isinstance(parsed, dict): ok_all = False; break
            if isinstance(parsed.get("bloco2"), list): agg["bloco2"] += [str(x) for x in parsed["bloco2"]]
            if isinstance(parsed.get("bloco3"), list): agg["bloco3"] += [str(x) for x in parsed["bloco3"]]
            if isinstance(parsed.get("rodape"), list): agg["rodape"] += [str(x) for x in parsed["rodape"]]
            if isinstance(parsed.get("bloco4"), dict):
                o = str(parsed["bloco4"].get("observacao","")).strip()
                c = str(parsed["bloco4"].get("conduta","")).strip()
                if o: agg["bloco4"]["observacao"] = o
                if c: agg["bloco4"]["conduta"] = c
        if not ok_all:
            continue
        def _dedup(seq: List[str]) -> List[str]:
            seen, out = set(), []
            for s in [clean_text(x) for x in seq if clean_text(x)]:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out
        agg["bloco2"] = _dedup(agg["bloco2"])[:30]
        agg["bloco3"] = _dedup(agg["bloco3"])[:30]
        agg["rodape"] = _dedup(agg["rodape"])[:30] or ["NENHUM ITEM SEM EVIDÊNCIA"]
        return (cfg["model"], agg)
    return (None, {})

# ---------------- Fallback Py para "relacionados à queixa" ----------------
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
    # dedup + limitar
    seen, uniq = set(), []
    for s in out:
        s2 = clean_text(s)
        if s2 and s2 not in seen:
            seen.add(s2); uniq.append(s2)
    return uniq[:30]

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Busca de prontuários + exames + análise Groq (JSON).")
    p.add_argument("-n", "--nome", help="Nome completo do paciente")
    p.add_argument("-d", "--nascimento", help="Data de nascimento dd/mm/yyyy")
    p.add_argument("-q", "--queixa", help="Queixa atual")
    p.add_argument("--desde", help="Buscar desde (dd/mm/yyyy). Se vazio, 12 meses.")
    p.add_argument("--like", action="store_true", help="Permitir LIKE no nome se igualdade não encontrar.")
    return p.parse_args()

def main():
    ensure_dirs()
    purge_old_files(JSON_DIR, 1)
    purge_old_files(REPORTS_DIR, 1)

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

    # -------- 1) CONSULTAS (primeiro decide recorte final) --------
    def _run_consultas(dt_inicio: date, like_flag: bool) -> pd.DataFrame:
        frames = []
        for lbl, conn_str in conns.items():
            df = query_posto(lbl, conn_str, nome, nasc_date, dt_inicio, use_like=like_flag)
            if not df.empty:
                df.insert(0, "posto", lbl)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df_all = _run_consultas(dt_ini_user, like_flag=False)
    if df_all.empty and (args.like or True):
        print("Nenhum dado com igualdade exata. Tentando busca parcial (LIKE)...")
        df_all = _run_consultas(dt_ini_user, like_flag=True)

    # Se <10 no recorte informado => recorte final vira 60 meses (para TUDO, inclusive exames)
    dt_ini_final = dt_ini_user
    if df_all.empty or len(df_all) < 10:
        print("Poucos registros no recorte. Ampliando busca para 60 meses…")
        dt_ini_5y = date.today().replace(day=1) - relativedelta(months=60)
        df_all_5y = _run_consultas(dt_ini_5y, like_flag=True)
        # Usamos SEMPRE o recorte final (5 anos) para todo o projeto
        dt_ini_final = dt_ini_5y
        df_all = df_all_5y

    # Normalização mínima das consultas
    if not df_all.empty:
        df_all.columns = [str(c).strip().lower() for c in df_all.columns]
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente","datainicioconsulta","idprontuario"]:
            if col not in df_all.columns: df_all[col] = None
        for col in ["queixa","observacao","conduta","especialidade","profissional_atendente"]:
            df_all[col] = df_all[col].map(clean_text)

    # ------ HISTÓRICO p/ IA (consultas) ------
    historicos = []
    if not df_all.empty:
        df_hist = df_all.copy()
        # remove "não compareceu/chamada" etc
        def _no_show_row(row: dict) -> bool:
            txt = " ".join([str(row.get(k,"")) for k in ("queixa","observacao","conduta")])
            txt_norm = strip_accents(txt).lower()
            return ("nao compareceu" in txt_norm) and ("chamad" in txt_norm)
        mask_send = []
        for _, row in df_hist.iterrows():
            mask_send.append(not _no_show_row(row.to_dict()))
        df_hist_send = df_hist.loc[mask_send].copy()
        # descarta registros totalmente em branco
        def _is_blank_series(s: pd.Series) -> pd.Series:
            if s is None: return pd.Series([True] * 0)
            return s.isna() | s.astype(str).str.strip().eq("") | s.astype(str).str.lower().eq("none")
        q_blank = _is_blank_series(df_hist_send.get("queixa", pd.Series([None]*len(df_hist_send))))
        c_blank = _is_blank_series(df_hist_send.get("conduta", pd.Series([None]*len(df_hist_send))))
        df_hist_send = df_hist_send.loc[~(q_blank & c_blank)].reset_index(drop=True)
        historicos = sanitize_for_json(df_hist_send).to_dict(orient="records")

    # -------- 2) EXAMES (depois de decidir dt_ini_final!) --------
    df_exams_all = query_exams_all_posts(conns, nome, nasc_date, dt_ini_final)
    exames_list_json = sanitize_for_json(df_exams_all).to_dict(orient="records") if not df_exams_all.empty else []
    # payload enxuto para prompt (limita a 200 e compacta alguns campos textuais)
    exams_compact_for_prompt: List[Dict[str, Any]] = []
    for r in exames_list_json[:200]:
        exams_compact_for_prompt.append(_compact_exam_row(r))

    # -------- 3) Bloco 1 (Py) --------
    if not df_all.empty:
        df_b1 = build_last10(df_all)
        b1_lines = df_b1["linha_fmt"].tolist()
    else:
        b1_lines = []
    bloco1_resumo_py = summarize_block1_lines(b1_lines, target=500, maxlen=1000)

    # -------- 4) Auditoria JSON bruto --------
    payload_full = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "data_inicial_utilizada": dt_ini_final.isoformat(),
            "criterio_tempo": "60_meses" if (dt_ini_final < dt_ini_user) else "desde_data_informada",
            "gerado_em": datetime.now().isoformat(),
            "registros_total_consultas": int(len(df_all)) if not isinstance(df_all, list) else 0,
            "registros_total_exames": int(len(df_exams_all)) if not isinstance(df_exams_all, list) else 0,
        },
        "consultas_amostra": sanitize_for_json(df_all).to_dict(orient="records") if not df_all.empty else [],
        "exames_amostra": exames_list_json,
    }
    json_path = save_json(payload_full)
    print(f"JSON: {os.path.basename(json_path)} (será limpo em ~1h)")

    # -------- 5) IA (Groq) --------
    prompt_txt = load_user_prompt()
    model_cfgs = _list_groq_model_configs()
    model_used, result_ia = call_groq_map_reduce(
        queixa, bloco1_resumo_py, historicos, model_cfgs, prompt_txt, exams_compact_for_prompt
    )
    titulo_b2 = "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL (IA)" if model_used else "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL (Py)"
    titulo_b3 = "DIAGNOSTICO DIFERENCIAL (IA)" if model_used else "DIAGNOSTICO DIFERENCIAL (Py)"
    titulo_b4 = "SUGESTAO CAMPOS OBS E CONDUTA (IA)" if model_used else "SUGESTAO CAMPOS OBS E CONDUTA (Py)"

    # Fallbacks
    bloco2 = result_ia.get("bloco2", []) if result_ia else []
    bloco3 = result_ia.get("bloco3", []) if result_ia else []
    bloco4 = result_ia.get("bloco4", {"observacao":"", "conduta":""}) if result_ia else {"observacao":"", "conduta":""}
    rodape = result_ia.get("rodape", ["NENHUM ITEM SEM EVIDÊNCIA"]) if result_ia else ["NENHUM ITEM SEM EVIDÊNCIA"]

    # 5.1) Se IA não trouxe relacionados, usa fallback Py
    if not bloco2:
        bloco2 = find_related_locally(df_all, queixa)
        titulo_b2 = "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL (Py)"
    if not bloco3:  # mantém vazio se preferir; aqui não forçamos diferencial Py
        pass

    # -------- 6) Saída FINAL --------
    output = {
        "ok": True,
        "paciente": nome,
        "data_nascimento": nasc_date.isoformat(),
        "queixa_atual": queixa,
        "registros_total": int(len(df_all)) if not df_all.empty else 0,
        "data_inicial": dt_ini_final.isoformat(),
        "TITULO_BLOCO1": "RESUMO DOS ÚLTIMOS ATENDIMENTOS (Py)",
        "BLOCO1_RESUMO": bloco1_resumo_py,  # <=1000 chars garantido
        "TITULO_BLOCO2": titulo_b2,
        "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": bloco2,
        "TITULO_BLOCO3": titulo_b3,
        "DIAGNOSTICO DIFERENCIAL": bloco3,
        "TITULO_BLOCO4": titulo_b4,
        "SUGESTAO CAMPOS OBS E CONDUTA": bloco4,
        "EXAMES_RESULTADOS": exames_list_json,     # JSON completo de exames
        "exames_csv": None,                         # se quiser, salve CSV depois com df_exams_all.to_csv(...)
        "rodape": rodape,
        "provider_mode": "json" if model_used else "py_fallback",
        "provider_used": model_used,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
