# Pasos para el articulo sobre el proyecto

## Creacion del ambiente de trabajo

Creacion del ambiente virutal

`python -m venv venv`

Activacion del ambiente virtual desde git bash

`source venv/Scripts/activate`

Instalacion de dependencias

`pip install streamlit langchain requests python-dotenv numpy`

Creacion del archivo requirements.txt

`pip freeze > requirements.txt`

Crear la interfaz principal

```
import streamlit as st

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

st.title("Welcome to the Webpage Gemini Assistant!")
```

### Creacion del Sidebar para ingresar el URL de la pagina web

```
with st.sidebar:
    st.header("Fuente de conocimiento")
    website_url = st.text_input("Ingresa la URL de tu sitio web...")
```

### Creacion del input

```
prompt = st.chat_input("Escribe tu pregunta...")
```

#### Ejemplo de interacciones

```
with st.chat_message("AI"):
    st.write("Hola, ¿en qué puedo ayudarte?")

with st.chat_message("User"):
    st.write("Dame información sobre la página web")
```

### Validacion de interacciones

```
prompt = st.chat_input("Escribe tu pregunta...")

if prompt is not None and prompt != "":
    with st.chat_message("User"):
        st.write(prompt)
    with st.chat_message("AI"):
        st.write("Respuesta por default.")
```

#### Validacion de interacciones

```
def get_response(prompt):
    return "Respuesta por default."

prompt = st.chat_input("Escribe tu pregunta...")

if prompt is not None and prompt != "":
    response = get_response(prompt)
    with st.chat_message("User"):
        st.write(prompt)
    with st.chat_message("AI"):
        st.write(response)
```

### Utilizando los esquemas LangChain

Importamos los esquemas de LangChain para trabajar con mensajes del asistente y los usuarios:

```
from langchain_core.messages import AIMessage, HumanMessage

prompt = st.chat_input("Escribe tu pregunta...")

if prompt is not None and prompt != "":
    response = get_response(prompt)
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=response))
```

### Agregando historial de interacciones

```
chat_history = [
    AIMessage(content="Hola, puedo ayudarte a conocer mas sobre tu sitio web.")
]

# Puedes ver el historial del chat en el Sidebar para validarlo:
with st.sidebar:
    st.write(chat_history)
```

Cada vez que ejecutas un evento en Streamlit, este carga nuevamente todo el codigo de Python, por lo que es necesario trabajar con una variable que almacene el historial de las conversaciones, esto se puede verificar revisando que el historial solo se actualiza una sola vez:

```
[
0:"AIMessage(content='Hola, puedo ayudarte a conocer mas sobre tu sitio web.', additional_kwargs={}, response_metadata={})"
1:"HumanMessage(content='Hola ', additional_kwargs={}, response_metadata={})"
2:"AIMessage(content='Respuesta por default.', additional_kwargs={}, response_metadata={})"
]
```

Para solucionar este problema, puedes usar el metodo st.session_state, podemos reemplazar el historial con el siguiente condicional:

```
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola, puedo ayudarte a conocer mas sobre tu sitio web.")
    ]
```

Y ahora se debe reemplazar la variable de `chat_history` por `st.session_state.chat_history`:

```
if prompt is not None and prompt != "":
    response = get_response(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response))
```

### Generar la conversación

Para mostrar el contenido de la conversacion historica en la interfaz usamos el siguiente codigo:

```
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
       with st.chat_message("AI"):
              st.write("Gemini:", message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write("Usuario:", message.content)
```

### Validacion del sidebar

La aplicacion solo deberia funcionar si es que se ingresa una URL en el sidebar, por lo que agregamos el siguiente condicional y hacemos que afecte a todo el codigo de los prompts y respuestas

```
if website_url is None or website_url == "":
    st.info("Por favor ingresa la URL de tu sitio web.")
else:

    # Prompts y respuestas
    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt is not None and prompt != "":
        response = get_response(prompt)
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write("Gemini:", message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write("Usuario:", message.content)
```

Ahora la conversacion con el asistente solo estara disponible cuando el usuario ingrese una URL.

## WebScrapping

Para poder extraer los datos de la página web necesitamos usar un paquete llamado BeautifulSoup, que nos permite scapear la información de cada sitio web mediante P tags.

```
pip install beautifulsoap4 langchain-community
```

Utilizamos `WebBaseLoader` para interpretar el contenido de la pagina web

```
from langchain_community.document_loaders import WebBaseLoader
...

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents
```

Modificar el codigo del condicional principal:

```
if website_url is None or website_url == "":
    st.info("Por favor ingresa la URL de tu sitio web.")
else:
    documents = get_vectorstore_from_url(website_url)
    with st.sidebar:
        st.write(documents)

    # Prompts y respuestas
    prompt = st.chat_input("Escribe tu pregunta...")

    if prompt is not None and prompt != "":
        response = get_response(prompt)
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write("Gemini:", message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write("Usuario:", message.content)
```

Luego, al insertar una pagina web, podemos ingresar el link de una pagina web estatica y visualizar los datos de texto extraidos.

### Division de texto con Langchain RecursiveCharacterTextSplitter

Para generar las diviciones de texto una vez extraido, utilizamos una herramienta de Langchain llamada RecursiveCharacterTextSplitter

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
...

# Modificar la funcion get_vectorstore_from_url

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split(document)

    return document_chunks
...

# Moficar el condicional para visualizar el sidebar

if website_url is None or website_url == "":
    st.info("Por favor ingresa la URL de tu sitio web.")
else:
    documents_chunks = get_vectorstore_from_url(website_url)
    with st.sidebar:
        st.write(documents_chunks)

```

Ahora al volver a enviar el link de la pagina web, obtendras el texto divido. Para ello utilizaremos una base de datos vectorial llama ChromaDB.

### Crear un vectorstore para los chunks

```
from langchain_community.vectorstores import Chroma
```

### Utiliza los Embeddings de Gemini en el VectorStore

Instala los paquetes de Langchain Gemini

```
pip install langchain-google-genai
```

Modifica el codigo de la aplicacion

```
from langchain_google_genai import GoogleGenerativeAIEmbeddings
...

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(document_chunks, embeddings=GoogleGenerativeAIEmbeddings())

    return vector_store
```

## Utiliza la API de Gemini

Al requerir los embeddings de Gemini, necesitamos una API Key de Gemini para poder usarlos.

### Genera el API Key

Ingresa a https://aistudio.google.com/app/apikey y genera una API Key de Gemini segun el nombre de proyecto de Google Cloud que vas a utilizar

Crea un archivo .env y coloca la clave de API, colocar el archivo .env en el Gitignore.

### Modifica la funcion get_vectorestore_from_url

Para incluir la validacion del api key de Gemini

```
from langchain.embeddings import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()
...

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key)

    vector_store = Chroma.from_documents(document_chunks, embeddings=embeddings)

    return vector_store
    ...
```
