from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
from qdrant_client import QdrantClient
import os
import logging
import json

# Configuração do logging para ver os dados recebidos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurações ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "services_qdrant") 
COLLECTION_NAME = "suporte_bling_v1"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# --- Validação de Configurações ---
if not OPENAI_API_KEY or not QDRANT_API_KEY:
    raise RuntimeError("As chaves de API OPENAI_API_KEY e QDRANT_API_KEY devem ser definidas.")

# --- Inicialização dos Clientes ---
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333, api_key=QDRANT_API_KEY, https=False)

app = FastAPI(
    title="ITCIA RAG API para Suporte Bling",
    description="Uma API que recebe perguntas de utilizadores, busca contexto na base de conhecimento do Bling (Qdrant) e gera respostas com um modelo de linguagem.",
    version="1.0.1"
)

def buscar_resposta_rag(query: str):
    try:
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding

        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=3
        )

        contexto = "\n\n---\n\n".join(
            [result.payload.get("resposta_estruturada", "") for result in search_results]
        )

        system_prompt = (
            "Você é o Nexo, um assistente de IA especialista no ERP Bling. Sua missão é responder às perguntas dos usuários de forma clara, "
            "objetiva e amigável, baseando-se estritamente no contexto fornecido. Se a resposta não estiver no contexto, "
            "informe educadamente que você não possui aquela informação específica, mas que pode ajudar com outras dúvidas sobre o Bling."
        )
        
        human_prompt = f"Com base no contexto abaixo, responda à seguinte pergunta do usuário.\n\nContexto:\n{contexto}\n\nPergunta do Usuário: {query}"

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ocorreu um erro no fluxo RAG: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar a sua pergunta.")

@app.get("/", tags=["Status"])
def healthcheck():
    return {"status": "ITCIA RAG API online!"}

@app.post("/webhook", tags=["Agente"])
async def webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"Webhook recebido: {json.dumps(data, indent=2)}") # Log para depuração
        
        # --- CORREÇÃO PRINCIPAL ---
        # O texto da mensagem de texto simples vem neste caminho.
        mensagem_usuario = data.get("data", {}).get("message", {}).get("conversation")
        
        if not mensagem_usuario:
            # Adiciona uma verificação para outros tipos de mensagem, se necessário no futuro
            logger.warning(f"O campo 'data.message.conversation' nao foi encontrado no payload.")
            # Retorna 200 OK para evitar que a Evolution API tente reenviar.
            return {"status": "Payload não continha mensagem de texto, ignorado."}

        resposta_ia = buscar_resposta_rag(mensagem_usuario)
        
        # Retorna a resposta no formato que a Evolution API espera
        return {"reply": resposta_ia}

    except json.JSONDecodeError:
        logger.error("Erro ao fazer o parse do JSON do webhook.")
        raise HTTPException(status_code=400, detail="Corpo da requisicao invalido, nao e um JSON.")
    except Exception as e:
        logger.error(f"Erro inesperado no webhook: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro ao processar o webhook.")


