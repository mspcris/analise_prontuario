import os
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from project import executar_pipeline_api  # importa seu m�dulo

class Req(BaseModel):
    nome: str
    data_nascimento: str  # mm/dd/yyyy
    queixa: str
    provedor: Literal["groq", "openai"]
    modelo: Optional[str] = None
    like: bool = False  # opcional: ativa busca parcial

    @field_validator("data_nascimento")
    def _valida_mmddyyyy(cls, v: str):
        import re
        if not re.fullmatch(r"(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}", v):
            raise ValueError("Use mm/dd/yyyy")
        return v

class Resp(BaseModel):
    ok: bool
    provedor: Optional[str] = None
    modelo: Optional[str] = None
    registros: Optional[int] = None
    json_path: Optional[str] = None
    analise: Optional[str] = None
    erro: Optional[str] = None

app = FastAPI(title="Backend Prontu�rios + LLM", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste em produ��o
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/analisar", response_model=Resp)
def analisar(body: Req):
    # sanity check de chaves
    if body.provedor == "groq" and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(500, "GROQ_API_KEY n�o configurada")
    if body.provedor == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY n�o configurada")

    result = executar_pipeline_api(
        nome=body.nome,
        data_nascimento_mmddyyyy=body.data_nascimento,
        queixa=body.queixa,
        provedor=body.provedor,
        modelo=body.modelo,
        like=body.like,
    )
    if not result.get("ok"):
        raise HTTPException(400, result.get("erro", "Falha no processamento"))
    return result
