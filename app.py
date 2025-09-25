from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
from qdrant_client import QdrantClient
import os

# --- Configurações ---
# Carrega as chaves a partir das variáveis de ambiente para segurança
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Usa o nome do serviço interno do Docker para uma conexão mais rápida e segura
QDRANT_HOST = os.getenv("QDRANT_HOST", "services_qdrant")
COLLECTION_NAME = "suporte_bling_v1"
EMBEDDING_MODEL = "text-embedding-3-small" # Modelo correto que usámos para carregar os dados
CHAT_MODEL = "gpt-4o-mini"

# --- Validação de Configurações ---
if not OPENAI_API_KEY or not QDRANT_API_KEY:
    raise RuntimeError("As chaves de API OPENAI_API_KEY e QDRANT_API_KEY devem ser definidas nas variáveis de ambiente.")

# --- Inicialização dos Clientes ---
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333, api_key=QDRANT_API_KEY)

app = FastAPI(
    title="ITCIA RAG API para Suporte Bling",
    description="Uma API que recebe perguntas de utilizadores, busca contexto na base de conhecimento do Bling (Qdrant) e gera respostas com um modelo de linguagem.",
    version="1.0.0"
)

def buscar_resposta_rag(query: str):
    """
    Executa o fluxo RAG (Retrieval-Augmented Generation).
    """
    try:
        # 1. Gerar embedding da pergunta do utilizador
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        ).data[0].embedding

        # 2. Buscar no Qdrant por vetores similares
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=3 # Pega os 3 resultados mais relevantes
        )

        # 3. Montar o contexto com base nos resultados
        contexto = "\n\n---\n\n".join(
            [result.payload.get("resposta_estruturada", "") for result in search_results]
        )

        # 4. Gerar a resposta final usando o modelo de chat com o contexto
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
            temperature=0.2 # Respostas mais diretas e menos criativas
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Ocorreu um erro no fluxo RAG: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar a sua pergunta.")

@app.get("/", tags=["Status"])
def healthcheck():
    """Verifica se a API está online."""
    return {"status": "ITCIA RAG API online!"}

@app.post("/webhook", tags=["Agente"])
async def webhook(request: Request):
    """
    Endpoint principal para receber as perguntas dos utilizadores (via Evolution API, por exemplo).
    """
    try:
        data = await request.json()
        
        # Adapte esta linha conforme o formato exato do payload da Evolution API
        mensagem_usuario = data.get("message", {}).get("text", "")
        
        if not mensagem_usuario:
            raise HTTPException(status_code=400, detail="O campo 'message.text' não foi encontrado no corpo da requisição.")

        resposta_ia = buscar_resposta_rag(mensagem_usuario)
        
        # O formato da resposta pode ser ajustado para o que a Evolution API espera
        return {"reply": resposta_ia}

    except Exception as e:
        print(f"Erro no webhook: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro ao processar o webhook.")

