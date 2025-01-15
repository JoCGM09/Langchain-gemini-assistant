import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever



import os
from dotenv import load_dotenv
load_dotenv()


def get_response(prompt):
    return "Respuesta por default."

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model="gemini-1.5-flash"
        )

    path_db = "/content/VectorDB" # @param {type:"string"}

    vector_store = Chroma(persist_directory=path_db)

    vector_store.add_documents(document_chunks, embeddings)

    return vector_store

def get_context_retriever_chain(vector_store):

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

# Configuración de la página
st.set_page_config(
    page_title="Webpage Gemini Assistant", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.test.com/help',
        'Report a bug': "https://www.test.com/bug",
        'About': "# Test *test* test"
    })

st.title("Bienvenido al asistente virtual de Gemini sobre tu página web")

# Historial del chat

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola, puedo ayudarte a conocer mas sobre tu sitio web.")
    ]

# Sidebar
with st.sidebar:
    st.header("Fuente de conocimiento")
    website_url = st.text_input("Ingresa la URL de tu sitio web...")

if website_url is None or website_url == "":
    st.info("Por favor ingresa la URL de tu sitio web.")
else:
    vector_store = get_vectorstore_from_url(website_url)
    
    retriever_chain = get_context_retriever_chain(vector_store)

    # Prompts y respuestas
    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt is not None and prompt != "":
        response = get_response(prompt)
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))

        retrieved_documents = retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": prompt
        })
        st.write(retrieved_documents)

     

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write("Gemini:", message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write("Usuario:", message.content)



