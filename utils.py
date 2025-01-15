import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_website_content(url):
    print("Descargando contenido de la página web...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Limpiar el contenido HTML
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        raise Exception(f"Error al descargar el contenido: {str(e)}")

def get_chat_model(context):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    system_prompt = f"""Eres un asistente experto que responde preguntas sobre el siguiente contenido web:

    {context}

    Instrucciones:
    1. Usa solo la información proporcionada arriba
    2. Da respuestas concisas y directas
    3. Si no encuentras la información específica, indícalo amablemente
    4. Mantén un tono conversacional pero profesional
    5. No inventes información que no esté en el contexto

    Responde a las preguntas del usuario basándote únicamente en este contexto."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=api_key,
        temperature=0.3,  # Temperatura más baja para respuestas más precisas
        top_p=0.8,
        top_k=30,
        max_output_tokens=1024
    )

    return llm, system_prompt 