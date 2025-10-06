# -*- coding: utf-8 -*-
import requests, json

URL = "http://127.0.0.1:8000/analisar"
payload = {
    "nome": "MARCELA SOUSA DE PAIVA",
    "data_nascimento": "09/17/1982",
    "queixa": "DOR DE CABEÇA",
    "provedor": "groq",
    "modelo": None,
    "like": False,
}

r = requests.post(URL, json=payload, timeout=120)
print("status:", r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
