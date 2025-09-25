# Usa uma imagem oficial do Python como base.
FROM python:3.11-slim

# Define o diretório de trabalho dentro do contentor.
WORKDIR /app

# Copia o ficheiro de dependências para o contentor.
COPY requirements.txt .

# Instala as dependências.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da sua aplicação para o contentor.
COPY app.py .

# Expõe a porta 8000 para que o Uvicorn possa correr nela.
EXPOSE 8000

# O comando que será executado quando o contentor iniciar.
# Ele inicia o servidor Uvicorn, que serve a sua aplicação FastAPI.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
