# -*- coding: utf-8 -*-
"""
project.py — Busca prontuários, monta BLOCO 1 local,
chama IA com failover (Groq<->OpenAI) e ENTREGA JSON no terminal.

Melhorias desta versão:
- Logs explícitos de tentativa/erro por provedor e modo (JSON e TEXTO).
- Força JSON com response_format quando suportado (OpenAI) e tenta emulada no Groq.
- Sanitiza saída da IA: remove cercas ``` e extrai o 1º objeto JSON válido.
- Retry exponencial por provedor.
- Reduz contexto de apoio (<=60 linhas) para reduzir alucinação e quedas.
- Nunca preenche 2/3/4 sem IA; se ambas falham, insere mensagem padrão.

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv groq requests
Chaves:
  GROQ_API_KEY, OPENAI_API_KEY (OPENAI_MODEL opcional; padrão gpt-4o-mini)
"""

import os, re, json, uuid, time, argparse, unicodedata, math
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Tuple, Iterable, Optional, Literal, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import requests

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

SQL_DIR      = os.path.join(BASE_DIR, "sql")
JSON_DIR     = os.path.join(BASE_DIR, "json")
PROMPTS_DIR  = os.path.join(BASE_DIR, "prompts")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")

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
        "p.idprontuario, p.paciente, p.datanascimento, "
        "p.queixa, p.observacao, p.conduta, "
        "p.datainicioconsulta, p.idespecialidade, p.especialidade, p.idmedico "
        "FROM cad_prontuario p"
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
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

# ---------------- Filtro "não compareceu" ----------------
def _no_show_row(row: dict) -> bool:
    txt = " ".join([str(row.get(k,"")) for k in ("queixa","observacao","conduta")])
    txt_norm = strip_accents(txt).lower()
    return ("nao compareceu" in txt_norm) and ("chamad" in txt_norm)

# ---------------- IA prompts ----------------
def _prompt_json(queixa_atual: str, bloco1_rows: List[Dict[str,Any]], apoio: List[Dict[str,Any]]) -> str:
    return (
        "Responda ESTRITAMENTE em JSON válido UTF-8, sem qualquer texto fora do JSON.\n"
        "REGRAS:\n"
        "1) PROIBIDO inventar. Só cite EXAMES/DIAGNÓSTICOS/MEDICAÇÕES do histórico.\n"
        "2) Sempre que citar algo clínico de um atendimento, ANEXE a FONTE no texto entre parênteses: "
        "(Posto {posto}, id {idprontuario}, data {datainicioconsulta}).\n"
        "3) Se não houver evidência, escreva exatamente: \"NÃO ENCONTRADO NO HISTÓRICO\".\n\n"
        f"QUEIXA_ATUAL: {queixa_atual}\n\n"
        "BLOCO1_FIXO (use exatamente estes itens, na ordem):\n"
        + json.dumps(bloco1_rows, ensure_ascii=False) + "\n\n"
        "APOIO (linhas recentes p/ cruzar nos Blocos 2/3/4):\n"
        + json.dumps(apoio, ensure_ascii=False) + "\n\n"
        'SAÍDA OBRIGATÓRIA (schema): {"bloco1":[{idprontuario,posto,data,resumo}], "bloco2":[string], "bloco3":[string], "bloco4":{...}, "rodape":[string]}'
    )

def _prompt_text(queixa_atual: str, bloco1_table_txt: str, apoio_txt: str) -> str:
    return (
        "Responda em TEXTO (NÃO JSON). Formato OBRIGATÓRIO:\n"
        "BLOCO 1\n<conteúdo>\n\nBLOCO 2\n<conteúdo>\n\nBLOCO 3\n<conteúdo>\n\nBLOCO 4\n<conteúdo>\n\nRODAPÉ\n<conteúdo>\n"
        "REGRAS:\n"
        "1) PROIBIDO inventar; cite só o que há no histórico.\n"
        "2) Sempre anexar FONTE: (Posto {posto}, id {idprontuario}, data {datainicioconsulta}).\n"
        '3) Sem evidência: "NÃO ENCONTRADO NO HISTÓRICO".\n'
        "4) No BLOCO 1, REPRODUZA EXATAMENTE a tabela fixa abaixo.\n\n"
        f"QUEIXA_ATUAL: {queixa_atual}\n\n"
        "BLOCO1_TABELA_FIXA:\n" + bloco1_table_txt + "\n\n"
        "APOIO (texto livre com até 60 linhas recentes):\n" + apoio_txt + "\n\n"
        "Retorne exatamente com os títulos BLOCO 1..RODAPÉ."
    )

# ---------------- IA calls + robustez ----------------
def _extract_json_candidate(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    # remove cercas ```...```
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL)
    # pega do primeiro { até o último } (heurística)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None

def _request_with_retries(func, label: str, attempts: int = 3, backoff: float = 1.5) -> Optional[str]:
    for i in range(attempts):
        out = None
        try:
            out = func()
        except Exception as e:
            print(f"[{label}] erro: {e}")
            out = None
        if out and out.strip():
            return out
        if i < attempts - 1:
            t = backoff ** i
            print(f"[{label}] vazio/sem resposta. retry em {t:.1f}s...")
            time.sleep(t)
    return None

def _groq_chat_raw(messages: List[Dict[str,str]], max_tokens=3000) -> Optional[str]:
    api_key = _env("GROQ_API_KEY","")
    if not api_key:
        print("[GROQ] Sem chave.")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=_env("GROQ_MODEL","llama-3.1-70b-versatile"),
            messages=messages,
            temperature=0.2,
            top_p=1,
            max_tokens=max_tokens
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[GROQ] exceção: {e}")
        return None

def _groq_chat_json(user_prompt: str) -> Optional[str]:
    print("[GROQ] tentando JSON...")
    def call():
        return _groq_chat_raw([
            {"role":"system","content":"Responda estritamente em JSON válido. Não use cercas de código."},
            {"role":"user","content": user_prompt}
        ])
    raw = _request_with_retries(call, "GROQ-JSON")
    if not raw:
        return None
    cand = _extract_json_candidate(raw)
    return cand or raw  # devolve bruto se heurística falhar

def _groq_chat_text(user_prompt: str) -> Optional[str]:
    print("[GROQ] tentando TEXTO...")
    def call():
        return _groq_chat_raw([
            {"role":"system","content":"Responda estritamente no formato pedido, sem explicações extras."},
            {"role":"user","content": user_prompt}
        ])
    return _request_with_retries(call, "GROQ-TEXT")

def _openai_chat_json(user_prompt: str) -> Optional[str]:
    print("[OPENAI] tentando JSON...")
    api_key = _env("OPENAI_API_KEY","")
    if not api_key:
        print("[OPENAI] Sem chave.")
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    body = {
        "model": _env("OPENAI_MODEL","gpt-4o-mini"),
        "messages": [
            {"role":"system","content":"Responda estritamente em JSON válido. Não use cercas de código."},
            {"role":"user","content": user_prompt}
        ],
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"}
    }
    def call():
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=90)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    raw = _request_with_retries(call, "OPENAI-JSON")
    if not raw:
        return None
    cand = _extract_json_candidate(raw)
    return cand or raw

def _openai_chat_text(user_prompt: str) -> Optional[str]:
    print("[OPENAI] tentando TEXTO...")
    api_key = _env("OPENAI_API_KEY","")
    if not api_key:
        print("[OPENAI] Sem chave.")
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    body = {
        "model": _env("OPENAI_MODEL","gpt-4o-mini"),
        "messages": [
            {"role":"system","content":"Responda estritamente no formato pedido, sem explicações extras."},
            {"role":"user","content": user_prompt}
        ],
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": 3000
    }
    def call():
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=90)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    return _request_with_retries(call, "OPENAI-TEXT")

# ---------------- Conversão TEXTO -> JSON local ----------------
def _split_blocks_text(raw: str) -> Dict[str,str]:
    txt = raw.replace("\r\n","\n")
    keys = ["BLOCO 1","BLOCO 2","BLOCO 3","BLOCO 4","RODAPÉ","RODAPE"]
    positions = {}
    for k in keys:
        m = re.search(rf"(?m)^\s*{k}\s*$", txt)
        if m: positions[k] = m.start()
    if not positions:
        return {}
    order = [k for k in ["BLOCO 1","BLOCO 2","BLOCO 3","BLOCO 4","RODAPÉ","RODAPE"] if k in positions]
    parts = {}
    for i, k in enumerate(order):
        start = positions[k]
        end = len(txt) if i == len(order)-1 else positions[order[i+1]]
        body = txt[start:end].split("\n",1)
        content = body[1] if len(body)>1 else ""
        parts[k] = content.strip()
    return parts

def _lines_nonempty(s: str) -> List[str]:
    return [ln.strip() for ln in (s or "").split("\n") if ln.strip()]

def _parse_bloco4_text(s: str) -> Dict[str, Any]:
    texto = s.strip()
    if not texto:
        return {"texto": ""}
    obs = []
    cond = []
    others = []
    for ln in _lines_nonempty(texto):
        ln_low = ln.lower()
        if ln_low.startswith("observação:") or ln_low.startswith("observacao:"):
            obs.append(ln.split(":",1)[1].strip() if ":" in ln else ln)
        elif ln_low.startswith("conduta:"):
            cond.append(ln.split(":",1)[1].strip() if ":" in ln else ln)
        else:
            others.append(ln)
    if obs or cond:
        out = {}
        if obs: out["observacao"] = " ".join(obs).strip()
        if cond: out["conduta"]   = " ".join(cond).strip()
        if others: out["observacoes_adicionais"] = " ".join(others).strip()
        return out
    return {"texto": texto}

def _render_b1_from_rows(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    for r in rows:
        out.append({
            "idprontuario": r.get("idprontuario"),
            "posto": r.get("posto"),
            "data": r.get("data"),
            "resumo": r.get("resumo")
        })
    return out

# ---------------- IA orquestração ----------------
def _try_json_then_text(queixa_atual: str,
                        bloco1_rows: List[Dict[str,Any]],
                        bloco1_table_txt: str,
                        apoio_rows: List[Dict[str,Any]],
                        apoio_txt: str,
                        primary: Literal["groq","openai"]="groq") -> Dict[str,Any]:

    # 1) JSON (primário -> secundário)
    json_prompt = _prompt_json(queixa_atual, bloco1_rows, apoio_rows)
    order = [primary, "openai" if primary=="groq" else "groq"]
    for prov in order:
        raw = _groq_chat_json(json_prompt) if prov=="groq" else _openai_chat_json(json_prompt)
        if not raw:
            print(f"[{prov.upper()}] JSON: sem resposta.")
            continue
        cand = _extract_json_candidate(raw) or raw
        try:
            doc = json.loads(cand)
            if all(k in doc for k in ("bloco1","bloco2","bloco3","bloco4","rodape")):
                print(f"[{prov.upper()}] JSON: OK.")
                return {
                    "mode": "json",
                    "provider": prov,
                    "bloco1": _render_b1_from_rows(bloco1_rows),
                    "bloco2": doc.get("bloco2") or [],
                    "bloco3": doc.get("bloco3") or [],
                    "bloco4": doc.get("bloco4") or {},
                    "rodape": doc.get("rodape") or []
                }
        except Exception as e:
            print(f"[{prov.upper()}] JSON parse falhou: {e}")

    # 2) TEXTO (primário -> secundário)
    text_prompt = _prompt_text(queixa_atual, bloco1_table_txt, apoio_txt)
    for prov in order:
        raw = _groq_chat_text(text_prompt) if prov=="groq" else _openai_chat_text(text_prompt)
        if not raw:
            print(f"[{prov.upper()}] TEXTO: sem resposta.")
            continue
        parts = _split_blocks_text(raw)
        if parts:
            print(f"[{prov.upper()}] TEXTO: OK (convertido).")
            b2_list = _lines_nonempty(parts.get("BLOCO 2",""))
            b3_list = _lines_nonempty(parts.get("BLOCO 3",""))
            b4_obj  = _parse_bloco4_text(parts.get("BLOCO 4",""))
            rodape_list = _lines_nonempty(parts.get("RODAPÉ","") or parts.get("RODAPE",""))
            return {
                "mode": "text->json",
                "provider": prov,
                "bloco1": _render_b1_from_rows(bloco1_rows),
                "bloco2": b2_list,
                "bloco3": b3_list,
                "bloco4": b4_obj,
                "rodape": rodape_list
            }

    # 3) Falha total
    fail = "não houve resposta da IA - Falha de comunicação. Tem internet?"
    print("[IA] Falha total (Groq e OpenAI).")
    return {
        "mode": "fail",
        "provider": None,
        "bloco1": _render_b1_from_rows(bloco1_rows),
        "bloco2": [fail],
        "bloco3": [fail],
        "bloco4": {"texto": fail},
        "rodape": ["Itens sem evidência textual explícita: NENHUM"]
    }

# ---------------- Bloco 1 ----------------
def build_last10(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["_dt_ini"] = pd.to_datetime(df["datainicioconsulta"], errors="coerce")
    df = df.loc[df["_dt_ini"].notna(),
                ["idprontuario","posto","_dt_ini","queixa","observacao","conduta"]].sort_values("_dt_ini", ascending=False)
    keep = []
    for _, r in df.iterrows():
        txt = " ".join([clean_text(r.get("queixa","")), clean_text(r.get("observacao","")), clean_text(r.get("conduta",""))])
        keep.append(not _no_show_row({"queixa":txt, "observacao":"","conduta":""}))
    df = df.loc[keep].head(10).copy()
    df["data"] = df["_dt_ini"].dt.strftime("%d/%m/%Y")
    def _mk_resumo(r):
        partes = [clean_text(r["queixa"]), clean_text(r["observacao"]), clean_text(r["conduta"])]
        partes = [p for p in partes if p]
        if not partes: return ""
        s = ". ".join(partes).strip(" .")
        return s + "."
    df["resumo"] = df.apply(_mk_resumo, axis=1)
    return df

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Busca de prontuários + IA (Groq/OpenAI) com failover e saída JSON.")
    p.add_argument("-n", "--nome", help="Nome completo do paciente")
    p.add_argument("-d", "--nascimento", help="Data de nascimento dd/mm/yyyy")
    p.add_argument("-q", "--queixa", help="Queixa atual")
    p.add_argument("-a", "--api", choices=["openai", "groq", "1", "2"], help="Provedor primário (default: groq)")
    p.add_argument("--like", action="store_true", help="Se não achar por igualdade, tenta LIKE automaticamente")
    p.add_argument("--no-delete-json", action="store_true", help="Não pergunta e mantém o JSON local (audit)")
    return p.parse_args()

def main():
    ensure_dirs()
    purge_old_files(JSON_DIR, 1)
    purge_old_files(REPORTS_DIR, 1)

    args = parse_args()

    nome = args.nome or input("1) Nome completo do paciente: ").strip()
    data_nasc_str = args.nascimento or input("2) Data de nascimento (dd/mm/yyyy): ").strip()
    queixa = args.queixa or input("3) Queixa atual do paciente: ").strip()
    api_choice = args.api or "groq"
    api_choice = {"1":"openai","2":"groq"}.get(api_choice, api_choice)
    provider = "groq" if api_choice=="groq" else "openai"

    try:
        nasc_date = datetime.strptime(data_nasc_str, "%d/%m/%Y").date()
    except ValueError:
        print(json.dumps({"ok": False, "erro": "Data inválida. Use dd/mm/yyyy."}, ensure_ascii=False, indent=2))
        return

    conns = build_conns_from_env()
    if not conns:
        print(json.dumps({"ok": False, "erro": "Nenhum posto configurado no .env."}, ensure_ascii=False, indent=2))
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
        out = {
            "ok": True,
            "paciente": nome,
            "data_nascimento": nasc_date.isoformat(),
            "queixa_atual": queixa,
            "registros_total": 0,
            "bloco1": [],
            "bloco2": ["Nenhum dado encontrado."],
            "bloco3": ["Nenhum dado encontrado."],
            "bloco4": {"texto": "Nenhum dado encontrado."},
            "rodape": ["Itens sem evidência textual explícita: NENHUM"]
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # Normalização
    df_all.columns = [str(c).strip().lower() for c in df_all.columns]
    for col in ["queixa", "observacao", "conduta", "especialidade", "datainicioconsulta","posto","idprontuario"]:
        if col not in df_all.columns:
            df_all[col] = None
    for col in ["queixa", "observacao", "conduta", "especialidade"]:
        df_all[col] = df_all[col].map(clean_text)

    # BLOCO 1 local
    df_b1 = build_last10(df_all)
    b1_rows = []
    for _, r in df_b1.iterrows():
        b1_rows.append({
            "idprontuario": to_python_scalar(r["idprontuario"]),
            "posto": r["posto"],
            "data": r["data"],
            "resumo": r["resumo"]
        })
    b1_table_txt = "idprontuario | posto | data | resumo\n" + "\n".join(
        [f"{x['idprontuario']} | {x['posto']} | {x['data']} | {x['resumo']}" for x in b1_rows]
    )

    # JSON histórico (p/ auditoria e limpeza)
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

    # APOIO (<=60 linhas)
    df_all["_dt_ini"] = pd.to_datetime(df_all["datainicioconsulta"], errors="coerce")
    apo = df_all.sort_values("_dt_ini", ascending=False).head(60)
    apoio_rows = []
    apoio_txt_lines = []
    for _, c in apo.iterrows():
        dt_iso = ""
        try:
            dtt = pd.to_datetime(c.get("datainicioconsulta"), errors="coerce")
            if not pd.isna(dtt): dt_iso = dtt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            dt_iso = ""
        apoio_rows.append({
            "idprontuario": c.get("idprontuario"),
            "posto": c.get("posto"),
            "datainicioconsulta": dt_iso,
            "especialidade": c.get("especialidade") or "—",
            "queixa": c.get("queixa") or "",
            "observacao": c.get("observacao") or "",
            "conduta": c.get("conduta") or ""
        })
        apoio_txt_lines.append(
            f"({c.get('posto','?')}, id {c.get('idprontuario','?')}, data {c.get('datainicioconsulta','?')}) "
            f"queixa={c.get('queixa','')}; obs={c.get('observacao','')}; conduta={c.get('conduta','')}"
        )
    apoio_txt = "\n".join(apoio_txt_lines)

    # --- IA com failover (JSON -> TEXTO) ---
    ia = _try_json_then_text(
        queixa_atual=queixa,
        bloco1_rows=b1_rows,
        bloco1_table_txt=b1_table_txt,
        apoio_rows=apoio_rows,
        apoio_txt=apoio_txt,
        primary=("groq" if provider=="groq" else "openai")
    )

    # --- JSON final para terminal ---
    out = {
        "ok": True,
        "paciente": nome,
        "data_nascimento": nasc_date.isoformat(),
        "queixa_atual": queixa,
        "registros_total": int(len(df_all)),
        "provider_mode": ia.get("mode"),
        "provider_used": ia.get("provider"),
        "bloco1": ia["bloco1"],
        "bloco2": ia["bloco2"],
        "bloco3": ia["bloco3"],
        "bloco4": ia["bloco4"],
        "rodape": ia["rodape"]
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # limpeza opcional imediata
    if args.no_delete_json:
        return
    try:
        opt = input("Deseja apagar o JSON gerado agora? (s/n): ").strip().lower()
    except Exception:
        opt = "n"
    if opt == "s":
        try:
            os.remove(json_path)
        except Exception:
            pass

# ---------------- Entry ----------------
if __name__ == "__main__":
    main()
