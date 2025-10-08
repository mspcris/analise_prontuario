# -*- coding: utf-8 -*-
"""
project.py — Busca prontuários multi-postos e gera análise via Groq (apenas).
- Lê PROMPT a partir de: ./prompts/prompt.txt
- Lê modelos/parametrizações a partir de ./groq_modelos/*.ini
  • Ordem de tentativa = prefixo numérico do arquivo (1-*, 2-*, 3-*, 4-*)
  • Cada .ini pode definir: model, temperature, top_p, max_tokens, reasoning_effort, json_mode, stop
- Envia o histórico em CHUNKS (map→reduce) para evitar 'context_length_exceeded'
- Saída no terminal: APENAS JSON

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv requests
"""

import os, re, json, uuid, time, argparse, unicodedata
from datetime import datetime
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
    def _mk_resumo(r):
        partes = [clean_text(r["queixa"]), clean_text(r["observacao"]), clean_text(r["conduta"])]
        partes = [p for p in partes if p]
        return (". ".join(partes).strip(" .") + ".") if partes else ""
    df["resumo"] = df.apply(_mk_resumo, axis=1)
    return df

# ---------------- Prompt ----------------
def _read_file_any_encoding(path: str) -> str:
    encs = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    # última tentativa binária
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def load_user_prompt() -> str:
    if not os.path.isfile(PROMPT_FILE):
        return (
            "# STRICT_EVIDENCE_PROMPT (fallback)\n"
            "Responda SOMENTE JSON com chaves: bloco2 (lista), bloco3 (lista), "
            "bloco4 {observacao,conduta}, rodape (lista). Não invente fatos."
        )
    return _read_file_any_encoding(PROMPT_FILE).strip()

def build_prompt_chunk(queixa_atual: str,
                       ultimos_atend: List[Dict[str, Any]],
                       historicos_chunk: List[Dict[str, Any]],
                       user_prompt: str,
                       part_idx: int,
                       part_total: int) -> str:
    # bloco1 compacto para referência (com especialidade)
    linhas_vis = ["idprontuario | posto | data | especialidade | resumo"]
    for r in ultimos_atend:
        linhas_vis.append(
            f'{r.get("idprontuario")} | {r.get("posto")} | {r.get("data")} | {r.get("especialidade","")} | {r.get("resumo","")}'
        )
    bloco1_txt = "\n".join(linhas_vis)
    hist_json = json.dumps(historicos_chunk, ensure_ascii=False)
    header = (
        f"QUEIXA_ATUAL:\n{queixa_atual}\n\n"
        f"ULTIMOS_ATENDIMENTOS (NÃO ALTERAR):\n{bloco1_txt}\n\n"
        f"HISTORICO_JSON (PARTE {part_idx}/{part_total}):\n{hist_json}\n\n"
    )
    return header + (user_prompt or "")

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
                if model:
                    cfg["model"] = model
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
            # manter apenas o primeiro arquivo de cada ordem
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

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    body: Dict[str, Any] = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "Você é um assistente clínico objetivo. Responda em pt-BR."},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.get("temperature", 0.1),
        "top_p": cfg.get("top_p", 1),
        "max_tokens": int(cfg.get("max_tokens", 1800)),
        "stream": False,
    }
    if cfg.get("json_mode", True):
        body["response_format"] = {"type": "json_object"}
    if cfg.get("stop"):
        body["stop"] = cfg["stop"]
    eff = (cfg.get("reasoning_effort") or "").strip()
    if eff:
        body["reasoning"] = {"effort": eff}

    label = f"GROQ-{cfg['model']}"
    print(f"[GROQ] tentando modelo: {cfg['model']}")

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

# ---------------- Map → Reduce (chunking) ----------------
def call_groq_map_reduce(queixa_atual: str,
                         ultimos_atend: List[Dict[str, Any]],
                         historicos: List[Dict[str, Any]],
                         model_cfgs: List[Dict[str, Any]],
                         user_prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Retorna (modelo_utilizado, resultado_json_dict) ou (None, vazio) em falha.
    """
    if not model_cfgs:
        return (None, {})

    CHUNK_MAX_ITEMS = 120
    parts = [historicos[i:i+CHUNK_MAX_ITEMS] for i in range(0, len(historicos), CHUNK_MAX_ITEMS)]
    total_parts = max(1, len(parts))

    for cfg in model_cfgs:
        agg_bloco2: List[str] = []
        agg_bloco3: List[str] = []
        agg_obs, agg_cond = "", ""
        agg_rodape: List[str] = []

        ok_all = True
        for idx, chunk in enumerate(parts, start=1):
            prompt = build_prompt_chunk(queixa_atual, ultimos_atend, chunk, user_prompt, idx, total_parts)
            out = groq_chat_json(prompt, cfg)
            if not out:
                ok_all = False
                break
            try:
                parsed = json.loads(out)
            except Exception:
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
    p = argparse.ArgumentParser(description="Busca de prontuários multi-postos + análise Groq (JSON).")
    p.add_argument("-n", "--nome", help="Nome completo do paciente")
    p.add_argument("-d", "--nascimento", help="Data de nascimento dd/mm/yyyy")
    p.add_argument("-q", "--queixa", help="Queixa atual")
    p.add_argument("--like", action="store_true", help="Se não achar por igualdade, tenta LIKE automaticamente")
    return p.parse_args()

def main():
    ensure_dirs()
    purge_old_files(JSON_DIR, 1)
    purge_old_files(REPORTS_DIR, 1)

    args = parse_args()
    nome = args.nome or input("1) Nome completo do paciente: ").strip()
    data_nasc_str = args.nascimento or input("2) Data de nascimento (dd/mm/yyyy): ").strip()
    queixa = args.queixa or input("3) Queixa atual do paciente: ").strip()

    try:
        nasc_date = datetime.strptime(data_nasc_str, "%d/%m/%Y").date()
    except ValueError:
        print(json.dumps({"ok": False, "erro": "Data inválida. Use dd/mm/yyyy."}, ensure_ascii=False))
        return

    conns = build_conns_from_env()
    if not conns:
        print(json.dumps({"ok": False, "erro": "Nenhum posto configurado no .env."}, ensure_ascii=False))
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
        print(json.dumps({
            "ok": True,
            "paciente": nome,
            "data_nascimento": nasc_date.isoformat(),
            "queixa_atual": queixa,
            "registros_total": 0,
            "provider_mode": "none",
            "provider_used": None,
            "ULTIMOS ATENDIMENTOS": [],
            "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": ["sem registros"],
            "DIAGNOSTICO DIFERENCIAL": ["sem registros"],
            "SUGESTAO CAMPOS OBS E CONDUTA": {"observacao": "", "conduta": ""},
            "rodape": ["NENHUM"]
        }, ensure_ascii=False, indent=2))
        return

    # Normalização mínima
    df_all.columns = [str(c).strip().lower() for c in df_all.columns]
    for col in ["queixa","observacao","conduta","especialidade","profissional_atendente","datainicioconsulta"]:
        if col not in df_all.columns:
            df_all[col] = None
    for col in ["queixa","observacao","conduta","especialidade","profissional_atendente"]:
        df_all[col] = df_all[col].map(clean_text)

    # ----- ULTIMOS ATENDIMENTOS -----
    df_b1 = build_last10(df_all)
    ultimos_atendimentos: List[Dict[str,Any]] = []
    for _, r in df_b1.iterrows():
        ultimos_atendimentos.append({
            "idprontuario": int(r["idprontuario"]) if pd.notna(r["idprontuario"]) else None,
            "posto": r.get("posto"),
            "data": r.get("data"),
            "especialidade": r.get("especialidade") or "",
            "resumo": r.get("resumo",""),
        })

    # ------ HISTÓRICO p/ IA ------
    df_hist = df_all.copy()
    mask_send = []
    for _, row in df_hist.iterrows():
        mask_send.append(not _no_show_row(row.to_dict()))
    df_hist_send = df_hist.loc[mask_send].copy()

    def _is_blank_series(s: pd.Series) -> pd.Series:
        if s is None:
            return pd.Series([True] * 0)
        return s.isna() | s.astype(str).str.strip().eq("") | s.astype(str).str.lower().eq("none")
    q_blank = _is_blank_series(df_hist_send.get("queixa", pd.Series([None] * len(df_hist_send))))
    c_blank = _is_blank_series(df_hist_send.get("conduta", pd.Series([None] * len(df_hist_send))))
    df_hist_send = df_hist_send.loc[~(q_blank & c_blank)].reset_index(drop=True)

    historicos = sanitize_for_json(df_hist_send).to_dict(orient="records")

    # salvar JSON “completo” (auditoria) — limpo em <=1h
    payload_full = {
        "metadata": {
            "paciente_input": nome,
            "data_nascimento_input": nasc_date.isoformat(),
            "queixa_input": queixa,
            "gerado_em": datetime.now().isoformat(),
            "registros_total": int(len(df_all))
        },
        "amostra": sanitize_for_json(df_all).to_dict(orient="records")
    }
    json_path = save_json(payload_full)
    print(f"JSON: {os.path.basename(json_path)} (será limpo em ~1h)")

    # ---------------- IA (Groq) ----------------
    prompt_txt = load_user_prompt()
    model_cfgs = _list_groq_model_configs()
    model_used, result = call_groq_map_reduce(queixa, ultimos_atendimentos, historicos, model_cfgs, prompt_txt)

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
        "registros_total": int(len(df_all)),
        "provider_mode": "json" if model_used else "fail",
        "provider_used": model_used,
        "ULTIMOS ATENDIMENTOS": ultimos_atendimentos,
        "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": result.get("bloco2", []),
        "DIAGNOSTICO DIFERENCIAL": result.get("bloco3", []),
        "SUGESTAO CAMPOS OBS E CONDUTA": result.get("bloco4", {"observacao":"", "conduta":""}),
        "rodape": result.get("rodape", ["NENHUM"])
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

# ---------------- Entry ----------------
if __name__ == "__main__":
    main()
