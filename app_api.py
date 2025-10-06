import os
import re
from typing import Literal
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from project import executar_pipeline_api

load_dotenv()


def _ddmmyyyy_to_mmddyyyy(s: str) -> str:
    d, m, y = s.split("/")
    return f"{m}/{d}/{y}"


class Req(BaseModel):
    nome: str
    data_nascimento: str  # dd/mm/aaaa
    queixa: str
    provedor: Literal["groq", "openai"]

    @field_validator("data_nascimento")
    @classmethod
    def _valida_data(cls, v: str) -> str:
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", v):
            raise ValueError("Use o formato dd/mm/aaaa")
        try:
            # valida data real, inclusive 29/02
            datetime.strptime(v, "%d/%m/%Y")
        except ValueError as e:
            raise ValueError(f"Data inválida: {e}")
        return v


app = FastAPI(title="Analise Prontuario API")

# CORS básico; ajuste ORIGINS se necessário (ex: "http://localhost:3000")
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/analisar")
def analisar(req: Req):
    try:
        data_us = _ddmmyyyy_to_mmddyyyy(req.data_nascimento)
        # Se o pipeline já aceita dd/mm/aaaa, troque para req.data_nascimento
        resultado = executar_pipeline_api(
            nome=req.nome,
            data_nascimento=data_us,
            queixa=req.queixa,
            provedor=req.provedor,
        )
        return {"ok": True, "resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"erro: {e}")
