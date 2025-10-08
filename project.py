# -*- coding: utf-8 -*-
"""
project.py — CLI para buscar prontuários em múltiplos bancos (por “posto”),
gerar JSON SOMENTE com históricos e enviar a análise para LLM (Groq) em JSON.

Principais recursos
- Lê prompt de prompts/prompt.txt (com fallback de encoding).
- Busca multi-postos (SQL custom por arquivo em ./sql/<POSTO>.sql).
- "ULTIMOS ATENDIMENTOS": sempre os 10 mais recentes (com especialidade).
- Filtra de ENVIAR À IA entradas com “não compareceu ... chamad”.
- Limpa ./json e ./reports >1h a cada execução.
- Groq: lê N modelos .ini em ./groq_modelos por ordem numérica (1-*.ini, 2-*.ini...).
  Campos suportados nos .ini (seção [groq]): model, temperature, top_p, max_tokens,
  reasoning_effort, json_mode, stop (lista JSON).
- Adaptive chunking: se der context_length_exceeded, reduz o HISTÓRICO e re-tenta.
- Nomes de chaves no JSON final:
  "ULTIMOS ATENDIMENTOS",
  "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL",
  "DIAGNOSTICO DIFERENCIAL",
  "SUGESTAO CAMPOS OBS E CONDUTA"

.env (exemplos para cada posto, A/N/X/...):
  DB_HOST_A=...
  DB_PORT_A=1433
  DB_BASE_A=...
  DB_USER_A=...
  DB_PASSWORD_A=...
  DB_ENCRYPT=yes
  DB_TRUST_CERT=yes
  DB_TIMEOUT=5
  GROQ_API_KEY=...

Requisitos:
  pip install "sqlalchemy>=2,<3" pyodbc pandas numpy python-dotenv requests
"""

import os, re, json, uuid, time, argparse, unicodedata
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Tuple, Iterable, Optional, Literal, Dict, Any

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
PROMPT_FILE  = os.path.join(PROMPTS_DIR, "prompt.txt")

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

def load_user_prompt() -> str:
    """Lê prompts/prompt.txt com tentativas de encoding."""
    paths = [PROMPT_FILE]
    for p in paths:
        if os.path.isfile(p):
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                try:
                    with open(p, "r", encoding=enc) as f:
                        return f.read().strip()
                except Exception:
                    continue
    return ""  # vazio não quebra; só reduz contexto

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
    # default genérico (ajuste ao seu schema, se necessário)
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
    cols = ["idprontuario","posto","_dt_ini","especialidade","queixa","observacao","conduta"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = (
        df.loc[df["_dt_ini"].notna(), cols]
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

# ---------------- Prompt base do sistema ----------------
STRICT_PROMPT = """# STRICT_EVIDENCE_PROMPT_v3 (pt-BR)

REGRAS OBRIGATÓRIAS
1) PROIBIDO inventar. Só cite EXAMES/DIAGNÓSTICOS/MEDICAÇÕES que estejam no HISTORICO_JSON.
2) Sempre que citar algo clínico do histórico, ANEXE a FONTE: (Posto {posto}, id {idprontuario}, data {datainicioconsulta}).
3) Se não houver evidência no JSON, escreva literalmente: “NÃO ENCONTRADO NO HISTÓRICO”.

SAÍDA OBRIGATÓRIA (JSON — SOMENTE JSON):
{
  "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": [
    "Linha por atendimento similar (>=70% de similaridade com a queixa atual), com RESUMO CONCISO que inclua: queixa, achados relevantes e CONDUTA registrada (se houver) + FONTE."
  ],
  "DIAGNOSTICO DIFERENCIAL": [
    "Hipóteses diagnósticas em linguagem médica (ex.: gastrite/DRGE/úlcera péptica/colelitíase/pancreatite/angina...), priorizadas por probabilidade e red flags; se citar algo do histórico, incluir FONTE."
  ],
  "SUGESTAO CAMPOS OBS E CONDUTA": {
    "observacao": "Texto clínico objetivo (estilo médico: queixa, evolução, sinais vitais/achados, fatores de risco/red flags quando houver; sem inventar).",
    "conduta": "Plano objetivo em linguagem médica (ex.: medidas de suporte, exames pertinentes, critérios de retorno e sinais de alarme; evitar prescrever fármacos sem evidência do histórico)."
  ],
  "rodape": [ "Itens sem evidência textual explícita: ... ou NENHUM" ]
}

RESTRIÇÕES
- NÃO repetir nem alterar a seção 'ULTIMOS_ATENDIMENTOS' (já fornecida).
- Use linguagem técnica, concisa e isenta.
- A resposta deve ser JSON válido (sem comentários/campos extras).
"""

def build_prompt_json(queixa_atual: str, bloco1_linhas: List[Dict[str, Any]], historicos_filtrados: List[Dict[str, Any]]) -> str:
    linhas_vis = ["idprontuario | posto | data | especialidade | resumo"]
    for r in bloco1_linhas:
        linhas_vis.append(f'{r["idprontuario"]} | {r["posto"]} | {r["data"]} | {r.get("especialidade","")} | {r["resumo"]}')
    bloco1_txt = "\n".join(linhas_vis)
    hist_json = json.dumps(historicos_filtrados, ensure_ascii=False)
    usr_prompt = load_user_prompt()
    prompt = (
        (usr_prompt + "\n\n") if usr_prompt else ""
    ) + (
        f"QUEIXA_ATUAL:\n{queixa_atual}\n\n"
        f"ULTIMOS_ATENDIMENTOS (NÃO ALTERAR):\n{bloco1_txt}\n\n"
        f"HISTORICO_JSON:\n{hist_json}\n\n"
        f"{STRICT_PROMPT}"
    )
    return prompt

# ---------------- Groq: carregar modelos .ini por ordem numérica ----------------
def _parse_ini(path: str) -> Optional[dict]:
    try:
        txt = open(path, "r", encoding="utf-8").read().replace("\r\n", "\n")
    except Exception:
        try:
            txt = open(path, "r", encoding="latin-1").read()
        except Exception:
            return None
    parser = ConfigParser(inline_comment_prefixes=("#",";"))
    try:
        parser.read_string(txt)
    except Exception:
        return None
    if not parser.has_section("groq"):
        return None
    g = dict(parser.items("groq"))
    cfg = {
        "model": g.get("model","").strip().strip('"').strip("'"),
        "temperature": float(g.get("temperature", "0.2") or 0.2),
        "top_p": float(g.get("top_p", "1") or 1),
        "max_tokens": int(g.get("max_tokens", "1800") or 1800),
        "json_mode": str(g.get("json_mode","true")).strip().lower() in {"1","true","yes","on"},
        "reasoning_effort": (g.get("reasoning_effort","") or "").strip().strip('"').strip("'"),
        "stop": None
    }
    if "stop" in g:
        try:
            cfg["stop"] = json.loads(g["stop"])
            if not isinstance(cfg["stop"], list):
                cfg["stop"] = None
        except Exception:
            cfg["stop"] = None
    if not cfg["model"]:
        return None
    return cfg

def load_groq_models_ordered() -> List[dict]:
    folder = os.path.join(BASE_DIR, "groq_modelos")
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".ini")]
    # ordenar pelo prefixo numérico; empatar por nome
    def keyfn(fname):
        m = re.match(r"^\s*(\d+)", fname)
        n = int(m.group(1)) if m else 9999
        return (n, fname.lower())
    files.sort(key=keyfn)
    out = []
    for fname in files:
        cfg = _parse_ini(os.path.join(folder, fname))
        if cfg:
            out.append(cfg)
    return out

# ---------------- Groq REST + adaptive chunking ----------------
def _request_with_retries(func, label: str, attempts: int = 3, backoff: float = 1.5) -> Optional[requests.Response]:
    import random
    resp = None
    for i in range(attempts):
        try:
            resp = func()
            return resp
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            txt = ""
            try:
                txt = e.response.json()
            except Exception:
                txt = getattr(e.response, "text", "")[:800]
            print(f"[{label}] HTTP {status}: {txt}")
            if i < attempts - 1:
                delay = (backoff ** i) + random.uniform(0,0.4)
                print(f"[{label}] retry em {delay:.1f}s...")
                time.sleep(delay)
                continue
            return None
        except Exception as e:
            print(f"[{label}] erro: {e}")
            if i < attempts - 1:
                delay = (backoff ** i)
                print(f"[{label}] retry em {delay:.1f}s...")
                time.sleep(delay)
                continue
            return None
    return resp

def groq_try_models_json(prompt_builder, queixa_atual, bloco1_list, hist_records) -> Tuple[Optional[str], Optional[str]]:
    """
    Itera modelos (ini) na ordem; para cada modelo faz adaptive chunking do HISTORICO
    quando há context_length_exceeded. Também remove 'reasoning' se for "unsupported".
    """
    api_key = _env("GROQ_API_KEY", "")
    if not api_key:
        print("[GROQ] falta GROQ_API_KEY.")
        return (None, None)

    models = load_groq_models_ordered()
    if not models:
        # fallback razoável
        models = [{
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.2, "top_p": 1.0, "max_tokens": 1800,
            "json_mode": True, "reasoning_effort": "", "stop": None
        }]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Começa com todo histórico; se estourar, corta pela metade (por nº de registros)
    records_sorted = hist_records[:]  # já saneado
    # ordenar por data de início se existir (string ISO)
    def _dtkey(r):
        d = r.get("datainicioconsulta")
        try:
            return d or ""
        except Exception:
            return ""
    records_sorted.sort(key=_dtkey, reverse=True)

    for cfg in models:
        model_name = cfg["model"]
        print(f"[GROQ] tentando modelo: {model_name}")

        # parâmetro reasoning (só se especificado)
        def _build_body(use_reasoning: bool, recs_slice: List[Dict[str,Any]]) -> dict:
            prompt = prompt_builder(queixa_atual, bloco1_list, recs_slice)
            body = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Você é um assistente clínico objetivo. Responda em pt-BR."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "max_tokens": int(cfg["max_tokens"]),
                "stream": False,
            }
            if cfg["json_mode"]:
                body["response_format"] = {"type": "json_object"}
            if cfg["stop"]:
                body["stop"] = cfg["stop"]
            if use_reasoning and cfg["reasoning_effort"]:
                body["reasoning"] = {"effort": cfg["reasoning_effort"]}
            return body

        # adaptive window (por número de registros)
        left = 1
        right = max(50, len(records_sorted))  # teto de busca binária leve
        right = min(right, len(records_sorted))
        best = None

        # primeiro, tenta com tudo
        try_count = 0
        use_reasoning_flag = True  # se der 'property reasoning unsupported', desliga e reenvia
        current_slice_size = len(records_sorted)

        while True:
            try_count += 1
            recs = records_sorted[:current_slice_size]
            body = _build_body(use_reasoning_flag, recs)

            def _call():
                resp = requests.post(url, headers=headers, json=body, timeout=90)
                if resp.status_code != 200:
                    resp.raise_for_status()
                return resp

            resp = _request_with_retries(_call, label=f"GROQ-{model_name}", attempts=3, backoff=1.3)
            if resp is None:
                break

            # Ok ou erro tratável
            try:
                data = resp.json()
            except Exception:
                data = {}

            # Checar estrutura
            if "choices" in data and data["choices"]:
                msg = (data["choices"][0].get("message") or {}).get("content", "") or ""
                if msg.strip():
                    return (model_name, msg.strip())

            # Se chegou aqui com 200 mas sem conteúdo, tenta reduzir
            if current_slice_size > 1:
                current_slice_size = max(1, current_slice_size // 2)
                continue
            break  # nada feito

        # Se caímos por exceptions HTTP, tentar analisar último erro textual do _request_with_retries
        # Não temos aqui, então refazemos uma tentativa "seca" só para ler o corpo de erro
        recs = records_sorted[:min(len(records_sorted), 200)]
        body_err = _build_body(use_reasoning_flag, recs)
        try:
            err = requests.post(url, headers=headers, json=body_err, timeout=60)
            if err.status_code == 400:
                try:
                    ej = err.json()
                except Exception:
                    ej = {}
                emsg = (ej.get("error") or {}).get("message", "").lower()
                # 1) property 'reasoning' is unsupported -> desliga e refaz com mesmo modelo
                if "reasoning" in emsg and "unsupported" in emsg and use_reasoning_flag:
                    use_reasoning_flag = False
                    # refaz com o modelo atual desde o começo
                    current_slice_size = len(records_sorted)
                    while True:
                        recs2 = records_sorted[:current_slice_size]
                        body2 = _build_body(False, recs2)
                        r2 = requests.post(url, headers=headers, json=body2, timeout=90)
                        if r2.status_code == 200:
                            j2 = r2.json()
                            if j2.get("choices"):
                                content = j2["choices"][0]["message"]["content"] or ""
                                if content.strip():
                                    return (model_name, content.strip())
                        else:
                            try:
                                jj = r2.json()
                            except Exception:
                                jj = {}
                            em = (jj.get("error") or {}).get("message","").lower()
                            if "context" in em and "length" in em:
                                if current_slice_size > 1:
                                    current_slice_size = max(1, current_slice_size // 2)
                                    continue
                        break
                # 2) context_length_exceeded -> reduz e re-tenta no mesmo modelo
                if "context" in emsg and "length" in emsg:
                    current_slice_size = min(len(records_sorted), max(1, len(records_sorted)//2))
                    while True:
                        recs3 = records_sorted[:current_slice_size]
                        body3 = _build_body(use_reasoning_flag, recs3)
                        r3 = requests.post(url, headers=headers, json=body3, timeout=90)
                        if r3.status_code == 200:
                            j3 = r3.json()
                            if j3.get("choices"):
                                content = j3["choices"][0]["message"]["content"] or ""
                                if content.strip():
                                    return (model_name, content.strip())
                        else:
                            try:
                                jj3 = r3.json()
                            except Exception:
                                jj3 = {}
                            em3 = (jj3.get("error") or {}).get("message","").lower()
                            if "context" in em3 and "length" in em3 and current_slice_size > 1:
                                current_slice_size = max(1, current_slice_size // 2)
                                continue
                        break
        except Exception:
            pass

    return (None, None)

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
    ultimos_atendimentos = []
    for _, r in df_b1.iterrows():
        ultimos_atendimentos.append({
            "idprontuario": int(r["idprontuario"]) if pd.notna(r["idprontuario"]) else None,
            "posto": r["posto"],
            "data": r["data"],
            "especialidade": r.get("especialidade") or "",
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

    # salvar JSON bruto (auditoria)
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

    # ---------------- IA (Groq com adaptive chunking) ----------------
    provider_used, ia_json_str = groq_try_models_json(
        prompt_builder=build_prompt_json,
        queixa_atual=queixa,
        bloco1_list=ultimos_atendimentos,
        hist_records=hist_json_records
    )

    # ---------------- Parse IA ----------------
    atds_rel, ddx, obscond, rodape = [], [], {"observacao": "", "conduta": ""}, []

    if ia_json_str:
        try:
            parsed = json.loads(ia_json_str)
            # aceitar nomes novos e também variantes antigas, se vierem
            b2 = parsed.get("ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL") or parsed.get("bloco2")
            b3 = parsed.get("DIAGNOSTICO DIFERENCIAL") or parsed.get("bloco3")
            b4 = parsed.get("SUGESTAO CAMPOS OBS E CONDUTA") or parsed.get("bloco4")
            rp = parsed.get("rodape") or []

            if isinstance(b2, list):
                atds_rel = [str(x) for x in b2]
            if isinstance(b3, list):
                ddx = [str(x) for x in b3]
            if isinstance(b4, dict):
                obscond = {
                    "observacao": str(b4.get("observacao","")),
                    "conduta": str(b4.get("conduta",""))
                }
            if isinstance(rp, list):
                rodape = [str(x) for x in rp]
        except Exception:
            ia_json_str = None

    if not ia_json_str:
        atds_rel = ["não houve resposta da IA - Falha de comunicação. Tem internet?"]
        ddx = ["não houve resposta da IA - Falha de comunicação. Tem internet?"]
        obscond = {"observacao": "não houve resposta da IA - Falha de comunicação. Tem internet?",
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
        "ULTIMOS ATENDIMENTOS": ultimos_atendimentos,
        "ATENDIMENTOS ANTERIORES RELACIONADOS A QUEIXA ATUAL": atds_rel,
        "DIAGNOSTICO DIFERENCIAL": ddx,
        "SUGESTAO CAMPOS OBS E CONDUTA": obscond,
        "rodape": rodape
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

# ---------------- Entry ----------------
if __name__ == "__main__":
    main()
