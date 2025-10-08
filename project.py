# -*- coding: utf-8 -*-
"""
project.py — CLI para buscar prontuários em múltiplos bancos (por “posto”),
gerar JSON SOMENTE com históricos e enviar a análise para LLM via Groq, em
modo map–reduce (chunking), com fail-over pela ordem numérica dos arquivos
em ./groq_modelos (1-*.txt, 2-*.txt, ...).

O prompt principal é lido de ./prompts/prompt.txt.

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv requests

.env (exemplos):
  DB_HOST_A=...
  DB_PORT_A=1433
  DB_BASE_A=...
  DB_USER_A=...
  DB_PASSWORD_A=...
  DB_ENCRYPT=yes
  DB_TRUST_CERT=yes
  DB_TIMEOUT=5

  GROQ_API_KEY=...

Estrutura esperada:
  /sql/{A..M}.sql            # opcional; se faltar, usa SELECT padrão
  /groq_modelos/1-*.txt      # modelo 1 (prioridade)
  /groq_modelos/2-*.txt      # modelo 2 (fallback), etc.
    - cada .txt pode ter:
        a) INI: [groq]\nmodel="nome-do-modelo"
        b) ou primeira linha útil com o nome do modelo

Saída do programa: APENAS JSON.
"""

import os, re, json, uuid, time, argparse, unicodedata
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from configparser import ConfigParser
import requests

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

SQL_DIR      = os.path.join(BASE_DIR, "sql")
JSON_DIR     = os.path.join(BASE_DIR, "json")
PROMPTS_DIR  = os.path.join(BASE_DIR, "prompts")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
PROMPT_TXT   = os.path.join(PROMPTS_DIR, "prompt.txt")

POSTOS = ["A", "N", "X", "Y", "B", "R", "P", "C", "D", "G", "I", "J", "M"]

# ---------------- Utils ----------------
def ensure_dirs():
    for d in (SQL_DIR, JSON_DIR, PROMPTS_DIR, REPORTS_DIR):
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

def load_prompt_text() -> str:
    try:
        with open(PROMPT_TXT, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        return txt
    except Exception:
        # fallback curto e seguro
        return (
            'Responda APENAS JSON válido com as chaves: '
            '{"bloco2":[...],"bloco3":[...],"bloco4":{"observacao":"","conduta":""},"rodape":[...]} '
            'Não invente. Cite fontes (Posto X, id Y, data Z) quando usar histórico.'
        )

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
    # default (ajuste conforme schema local)
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

# ---------------- “Não compareceu” filtro ----------------
def _no_show_row(row: dict) -> bool:
    txt = " ".join([str(row.get(k, "")) for k in ("queixa", "observacao", "conduta")])
    txt_norm = strip_accents(txt).lower()
    return ("nao compareceu" in txt_norm) and ("chamad" in txt_norm)

# ---------------- Bloco 1 (últimos 10) ----------------
def build_last10(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["_dt_ini"] = pd.to_datetime(df["datainicioconsulta"], errors="coerce")
    df = (
        df.loc[df["_dt_ini"].notna(),
               ["idprontuario", "posto", "_dt_ini", "queixa", "observacao", "conduta"]]
          .sort_values("_dt_ini", ascending=False)
          .head(10)
          .copy()
    )
    df["data"] = df["_dt_ini"].dt.strftime("%d/%m/%Y")
    def _mk_resumo(r):
        partes = [clean_text(r["queixa"]), clean_text(r["observacao"]), clean_text(r["conduta"])]
        partes = [p for p in partes if p]
        if not partes:
            return ""
        s = ". ".join(partes).strip(" .")
        return s + "."
    df["resumo"] = df.apply(_mk_resumo, axis=1)
    return df

# ---------------- Chunking (map–reduce) ----------------
def _rough_tokens(s: str) -> int:
    # Estimativa simples (~4 chars por token)
    return max(1, len(s) // 4)

def _records_to_json(records):
    return json.dumps(records, ensure_ascii=False)

# prompt parcial (MAP) curto para extrair sinalizações + similares
PARTIAL_PROMPT = """Responda SOMENTE JSON no formato:
{
 "bloco2": ["..."],    // linhas com (Posto X, id Y, data Z) quando houver
 "sinais": ["..."],    // termos/achados relevantes extraídos do lote
 "red_flags": ["..."]  // red flags do lote
}
Sem repetir Bloco 1. Sem inventar. Use as fontes (Posto/id/data) dos itens DESTE lote quando aplicável.
"""

def _map_prompt(queixa_atual: str, lote_json: str) -> str:
    return f"QUEIXA_ATUAL:\n{queixa_atual}\n\nHISTORICO_JSON:\n{lote_json}\n\n{PARTIAL_PROMPT}"

def _reduce_prompt(queixa_atual: str, bloco1_linhas: list, parciais: list[dict]) -> str:
    linhas_vis = ['idprontuario | posto | data | resumo'] + [
        f'{r["idprontuario"]} | {r["posto"]} | {r["data"]} | {r["resumo"]}' for r in bloco1_linhas
    ]
    b1 = "\n".join(linhas_vis)
    agregado = {
        "bloco2_all": sum((p.get("bloco2",[]) for p in parciais), []),
        "sinais_all": sum((p.get("sinais",[]) for p in parciais), []),
        "red_flags_all": sum((p.get("red_flags",[]) for p in parciais), []),
    }
    strict = load_prompt_text()  # conteúdo de prompts/prompt.txt
    return (
        f"QUEIXA_ATUAL:\n{queixa_atual}\n\n"
        f"BLOCO_1_TABELA_FIXA (NÃO ALTERAR):\n{b1}\n\n"
        f"PARCIAIS_JSON:\n{json.dumps(agregado, ensure_ascii=False)}\n\n"
        f"{strict}"
    )

def _chunk_records_for_model(records: List[dict], model_name: str, tgt_tokens_per_chunk: int) -> List[List[dict]]:
    """
    Divide o histórico em lotes que caibam no contexto alvo do modelo.
    Trunca campos textuais gigantes (proteção).
    """
    chunks, cur = [], []
    for r in sorted(records, key=lambda x: x.get("datainicioconsulta") or "", reverse=True):
        # truncagens defensivas
        r = dict(r)
        for k in ("queixa", "observacao", "conduta"):
            v = r.get(k) or ""
            if isinstance(v, str) and len(v) > 1200:
                r[k] = v[:1200] + "..."
        probe = _records_to_json(cur + [r])
        if _rough_tokens(probe) <= max(100, tgt_tokens_per_chunk):
            cur.append(r)
        else:
            if cur:
                chunks.append(cur)
            cur = [r]
    if cur:
        chunks.append(cur)
    return chunks

# ---------------- Groq REST ----------------
def _request_with_retries(func, label: str, attempts: int = 3, backoff: float = 1.5) -> Optional[str]:
    import random
    for i in range(attempts):
        try:
            out = func()
            if out and out.strip():
                return out
        except requests.HTTPError as e:
            retry_after = 0.0
            if e.response is not None:
                ra = e.response.headers.get("Retry-After")
                try:
                    retry_after = float(ra)
                except Exception:
                    retry_after = 0.0
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text if e.response is not None else ""
            print(f"[{label}] HTTP {getattr(e.response,'status_code',None)}: {str(body)[:800]}")
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

def _groq_chat_with_model(model: str, prompt: str, max_tokens: int = 900) -> Optional[str]:
    api_key = _env("GROQ_API_KEY","")
    if not api_key:
        print("[GROQ] falta GROQ_API_KEY.")
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    body = {
        "model": model,
        "messages": [
            {"role":"system","content":"Você é um assistente clínico objetivo. Responda em pt-BR."},
            {"role":"user","content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": int(max_tokens),
        "response_format": {"type":"json_object"},
        "stream": False
    }
    def _do():
        r = requests.post(url, headers=headers, json=body, timeout=90)
        if r.status_code != 200:
            try: print(f"[GROQ-{model}] HTTP {r.status_code}: {r.json()}")
            except: print(f"[GROQ-{model}] HTTP {r.status_code}: {r.text[:800]}")
            r.raise_for_status()
        data = r.json()
        choices = (data or {}).get("choices") or []
        if not choices or not choices[0].get("message", {}).get("content"):
            raise RuntimeError("groq_empty_or_malformed")
        return choices[0]["message"]["content"].strip()
    return _request_with_retries(_do, f"GROQ-{model}", attempts=3, backoff=1.5)

# ---------------- Modelos (ordem 1-,2-,3-...) ----------------
def _load_groq_models_ordered() -> List[str]:
    folder = os.path.join(BASE_DIR, "groq_modelos")
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if re.match(r"^\d+-.*\.txt$", f)]
    files.sort(key=lambda x: int(x.split("-")[0]))
    models = []
    for f in files:
        p = os.path.join(folder, f)
        try:
            txt = open(p,"r",encoding="utf-8").read()
            # tenta INI
            m = None
            try:
                cp = ConfigParser(inline_comment_prefixes=("#",";"))
                cp.read_string(txt)
                if cp.has_section("groq") and cp.has_option("groq","model"):
                    m = cp.get("groq","model").strip().strip('"').strip("'")
            except Exception:
                pass
            if not m:
                for ln in txt.splitlines():
                    ln = ln.strip()
                    if ln and not ln.startswith(("#",";","//","[")):
                        m = re.split(r"\s[#;].*$", ln)[0].strip().strip('"').strip("'")
                        break
            if m:
                models.append(m)
        except Exception:
            continue
    return models

# Limites conservadores por modelo (tokens de contexto)
CONTEXT_LIMITS = {
    "openai/gpt-oss-120b": 8000,          # modelo OSS 120b (8k aprox)
    "llama-3.3-70b-versatile": 128000,    # 128k
    "deepseek-r1-distill-llama-70b": 128000,
    "gemma2-9b-it": 8000,
}
RESERVED_COMPLETION_TOKENS = 900  # reserva para a resposta

# ---------------- IA Orquestração (map–reduce + fail-over) ----------------
def run_map_reduce_groq(queixa: str, bloco1_list: List[Dict[str,Any]], hist_json_records: List[Dict[str,Any]]) -> tuple[Optional[str], Optional[str]]:
    models_ordered = _load_groq_models_ordered() or ["llama-3.3-70b-versatile"]
    ia_json_str, provider_used = None, None

    for model in models_ordered:
        print(f"[GROQ] tentando modelo: {model}")
        max_ctx = CONTEXT_LIMITS.get(model, 32000)
        tgt_tokens_per_chunk = max(1000, max_ctx - RESERVED_COMPLETION_TOKENS - 1200)  # margem p/ instruções

        # --- MAP: dividir em lotes
        lotes = _chunk_records_for_model(hist_json_records, model, tgt_tokens_per_chunk)
        parciais = []
        ok_map = True

        for i, lote in enumerate(lotes, 1):
            p_map = _map_prompt(queixa, _records_to_json(lote))
            out = _groq_chat_with_model(model, p_map, max_tokens=min(800, RESERVED_COMPLETION_TOKENS))
            if not out:
                ok_map = False
                break
            try:
                parciais.append(json.loads(out))
            except Exception:
                ok_map = False
                break

        if not ok_map or not parciais:
            continue  # tenta próximo modelo

        # --- REDUCE: consolidar no formato final exigido em prompt.txt
        p_reduce = _reduce_prompt(queixa, bloco1_list, parciais)
        out_final = _groq_chat_with_model(model, p_reduce, max_tokens=min(1000, RESERVED_COMPLETION_TOKENS))
        if out_final:
            ia_json_str, provider_used = out_final, "groq"
            break

    return ia_json_str, provider_used

# ---------------- IA prompt builder (para redução final usa prompt.txt) ----------------
def build_prompt_json(queixa_atual: str, bloco1_linhas: List[Dict[str, Any]], historicos_filtrados: List[Dict[str, Any]]) -> str:
    # (mantido por compatibilidade, não usamos diretamente no map–reduce)
    linhas_vis = ["idprontuario | posto | data | resumo"]
    for r in bloco1_linhas:
        linhas_vis.append(f'{r["idprontuario"]} | {r["posto"]} | {r["data"]} | {r["resumo"]}')
    bloco1_txt = "\n".join(linhas_vis)
    hist_json = json.dumps(historicos_filtrados, ensure_ascii=False)
    prompt_main = load_prompt_text()
    return (
        f"QUEIXA_ATUAL:\n{queixa_atual}\n\n"
        f"BLOCO_1_TABELA_FIXA (NÃO ALTERAR):\n{bloco1_txt}\n\n"
        f"HISTORICO_JSON:\n{hist_json}\n\n"
        f"{prompt_main}"
    )

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Busca de prontuários multi-postos + análise Groq (JSON, chunking).")
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
            "bloco1": [],
            "bloco2": ["sem registros"],
            "bloco3": ["sem registros"],
            "bloco4": {"observacao": "", "conduta": ""},
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

    # ----- BLOCO 1 (últimos 10) -----
    df_b1 = build_last10(df_all)
    bloco1_list = []
    for _, r in df_b1.iterrows():
        bloco1_list.append({
            "idprontuario": int(r["idprontuario"]) if pd.notna(r["idprontuario"]) else None,
            "posto": r["posto"],
            "data": r["data"],
            "resumo": r["resumo"],
        })

    # ------ HISTÓRICO p/ IA (com filtros) ------
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

    hist_json_records = sanitize_for_json(df_hist_send).to_dict(orient="records")

    # salvar JSON “completo” (auditoria)
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

    # ---------------- IA (Groq map–reduce + fail-over) ----------------
    ia_json_str, provider_used = run_map_reduce_groq(queixa, bloco1_list, hist_json_records)

    bloco2, bloco3, bloco4, rodape = [], [], {"observacao": "", "conduta": ""}, []

    if ia_json_str:
        try:
            parsed = json.loads(ia_json_str)
            if isinstance(parsed.get("bloco2"), list):
                bloco2 = [str(x) for x in parsed["bloco2"]]
            if isinstance(parsed.get("bloco3"), list):
                bloco3 = [str(x) for x in parsed["bloco3"]]
            if isinstance(parsed.get("bloco4"), dict):
                bloco4 = {
                    "observacao": str(parsed["bloco4"].get("observacao","")),
                    "conduta": str(parsed["bloco4"].get("conduta",""))
                }
            if isinstance(parsed.get("rodape"), list):
                rodape = [str(x) for x in parsed["rodape"]]
        except Exception:
            provider_used = None
            ia_json_str = None

    if not ia_json_str:
        bloco2 = ["não houve resposta da IA - Falha de comunicação. Tem internet?"]
        bloco3 = ["não houve resposta da IA - Falha de comunicação. Tem internet?"]
        bloco4 = {"observacao": "não houve resposta da IA - Falha de comunicação. Tem internet?",
                  "conduta": "não houve resposta da IA - Falha de comunicação. Tem internet?"}
        if not rodape:
            rodape = ["Itens sem evidência textual explícita: NENHUM"]

    # ---------------- Saída FINAL ----------------
    output = {
        "ok": True,
        "paciente": nome,
        "data_nascimento": nasc_date.isoformat(),
        "queixa_atual": queixa,
        "registros_total": int(len(df_all)),
        "provider_mode": "json" if ia_json_str else "fail",
        "provider_used": provider_used,
        "bloco1": bloco1_list,
        "bloco2": bloco2,
        "bloco3": bloco3,
        "bloco4": bloco4,
        "rodape": rodape
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

# ---------------- Entry ----------------
if __name__ == "__main__":
    main()
