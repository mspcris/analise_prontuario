FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl gnupg2 apt-transport-https software-properties-common \
    unixodbc unixodbc-dev build-essential python3 python3-pip python3-venv

# Microsoft repo (tem o msodbcsql17)
RUN mkdir -p /usr/share/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" > /etc/apt/sources.list.d/microsoft-prod.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Python deps
RUN python3 -m venv .venv && . .venv/bin/activate && \
    pip install --upgrade pip && pip install -r requirements.txt

# Força driver 17 no código (ignora erro se não existir a string)
RUN sed -i "s/ODBC Driver 18 for SQL Server/ODBC Driver 17 for SQL Server/g" project.py || true

ENV PATH="/app/.venv/bin:${PATH}" \
    DB_ENCRYPT=yes \
    DB_TRUST_CERT=yes

EXPOSE 8000
CMD ["uvicorn","app_api:app","--host","0.0.0.0","--port","8000"]
