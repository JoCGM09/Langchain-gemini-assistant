
####################################################################################

#### ARCHIVO DE PRUEBA: EMBEDDINGS Y MODELO DE LENGUAJE GENERATIVO DE GOOGLE #######

####################################################################################



from langchain_core.messages import AIMessage, HumanMessage 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# URL que queremos analizar (parámetro fijo)
WEBSITE_URL = "https://www.cgsa.com.ec/certificaciones/"  # Cambia esta URL por la que desees analizar

def get_vectorstore_from_url(url):
    print("Iniciando carga de la página web...")
    loader = WebBaseLoader(
        url,
        header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    try:
        print("Descargando contenido...")
        document = loader.load()
        
        if not document:
            raise ValueError("No se pudo obtener contenido de la URL")
        print("Contenido descargado exitosamente")

        print("Procesando texto...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        document_chunks = text_splitter.split_documents(document)
        print(f"Texto dividido en {len(document_chunks)} fragmentos")

        print("Configurando embeddings...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/embedding-001",
            task_type="retrieval_document"
        )

        print("Creando vector store...")
        path_db = "./content/VectorDB"
        os.makedirs(path_db, exist_ok=True)

        vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=embeddings,
            persist_directory=path_db
        )
        print("Vector store creado exitosamente")
        
        return vector_store
    except Exception as e:
        print(f"Error detallado: {str(e)}")
        raise Exception(f"Error al procesar la URL: {str(e)}")

def get_conversation_chain(vector_store):
    print("Configurando modelo de lenguaje...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=api_key,
        temperature=0.9,
        top_p=0.9,
        top_k=40,
        max_output_tokens=2048
    )

    print("Configurando retriever...")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5
        }
    )

    print("Creando cadena de conversación...")
    
    prompt_template = """
    Eres un asistente experto y servicial especializado en Contecon Guayaquil S.A. (CGSA).
    Tu objetivo es proporcionar información útil incluso cuando las preguntas no son completamente precisas.

    Instrucciones específicas:
    1. Analiza el contexto de forma flexible y busca conexiones relevantes
    2. Si la pregunta no es exacta, intenta entender la intención del usuario
    3. Si encuentras información parcialmente relacionada, úsala para dar una respuesta útil
    4. Cuando sea posible, proporciona información adicional relevante
    5. Si no estás seguro, indica qué información sí tienes disponible
    6. Usa un tono conversacional y amigable
    7. Si la pregunta es ambigua, responde con las diferentes posibilidades

    Contexto disponible: {context}
    Pregunta del usuario: {question}

    Instrucciones adicionales:
    - Intenta siempre dar una respuesta útil
    - Si la información está parcialmente disponible, menciona lo que sabes
    - Sugiere preguntas relacionadas si es apropiado
    - Si necesitas más claridad, indica qué aspectos podrían ser útiles especificar

    Respuesta:
    """

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["question", "context"]
            )
        }
    )
    
    print("Cadena de conversación creada exitosamente")
    return conversation_chain

def main():
    chat_history = []
    print("Cargando información del sitio web...")
    print(f"URL: {WEBSITE_URL}")
    
    try:
        vector_store = get_vectorstore_from_url(WEBSITE_URL)
        conversation_chain = get_conversation_chain(vector_store)
        print("¡Información cargada! ¿Qué deseas saber?")

        while True:
            user_input = input("\nPregunta (o 'salir' para terminar): ")
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break

            result = conversation_chain.invoke({
                "question": user_input,
                "chat_history": chat_history
            })

            print("\nRespuesta:", result['answer'])
            chat_history.append((user_input, result['answer']))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()