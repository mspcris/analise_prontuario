import os
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from project import executar_pipeline_api  # importa seu m�dulo
from dotenv import load_dotenv
load_dotenv()  # carrega variáveis de .env para os.getenv(...)
from datetime import datetime

from pydantic import BaseModel, field_validator
import re

class Req(BaseModel):
    nome: str
    data_nascimento: str  # dd/mm/aaaa
    queixa: str
    provedor: Literal["groq", "openai"]
    modelo: Optional[str] = None
    like: bool = False

    @field_validator("data_nascimento")
    def _valida_ddmmyyyy(cls, v: str):
        v = v.strip()
        # 01-31 / 01-12 / 4 dígitos
        if not re.fullmatch(r"(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}", v):
            raise ValueError("Use dd/mm/aaaa")
        return v


class Resp(BaseModel):
    ok: bool
    provedor: Optional[str] = None
    modelo: Optional[str] = None
    registros: Optional[int] = None
    json_path: Optional[str] = None
    analise: Optional[str] = None
    erro: Optional[str] = None

app = FastAPI(title="Backend Prontuarios + LLM", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste em producao
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)



@app.post("/analisar", response_model=Resp)
def analisar(body: Req):
    # sanity check de chaves
    if body.provedor == "groq" and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(500, "GROQ_API_KEY nao configurada")
    if body.provedor == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY nao configurada")

    # CONVERSÃO dd/mm/aaaa -> mm/dd/yyyy para o pipeline legado
    try:
        dt = datetime.strptime(body.data_nascimento.strip(), "%d/%m/%Y")
        data_mmddyyyy = dt.strftime("%m/%d/%Y")
    except ValueError:
        raise HTTPException(422, "Use dd/mm/aaaa")

    result = executar_pipeline_api(
        nome=body.nome,
        data_nascimento_mmddyyyy=data_mmddyyyy,
        queixa=body.queixa,
        provedor=body.provedor,
        modelo=body.modelo,
        like=body.like,
    )
    if not result.get("ok"):
        raise HTTPException(400, result.get("erro", "Falha no processamento"))
    return result


@app.get("/health")
def health():
    keys = ["GROQ_API_KEY", "OPENAI_API_KEY"]
    env_ok = {k: bool(os.getenv(k)) for k in keys}
    db_vars = sorted([k for k in os.environ.keys() if k.startswith("DB_")])
    return {"ok": True, "env": env_ok, "db_vars": db_vars}
