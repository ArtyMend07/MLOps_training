FROM python:3.12-slim

WORKDIR /app

# Instala ferramentas do sistema necessarias e o gerenciador uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

# Copia manifestos de dependencia
COPY pyproject.toml uv.lock ./

# Sincroniza o ambiente isolado do projeto
RUN uv sync --frozen

# Copia o restante do codigo
COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
