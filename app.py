import streamlit as st
from utils import get_website_content, get_chat_model

# Configuración de la página
st.set_page_config(
    page_title="Chat con Website",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    /* Tema oscuro general */
    .stApp {
        background-color: #343541;
        color: #FFFFFF;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #202123;
    }
    
    /* Contenedor del chat */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Mensajes */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem auto;
        border-radius: 0.5rem;
        display: flex;
        max-width: 800px;
    }
    
    .user-message {
        background-color: #343541;
        border: 1px solid #565869;
    }
    
    .assistant-message {
        background-color: #444654;
    }
    
    /* Input */
    .stTextInput input {
        background-color: #40414F;
        border-color: #565869;
        border-radius: 0.5rem;
        color: white;
        font-size: 1rem;
        padding: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #565869;
        box-shadow: none;
    }
    
    /* Botones y elementos de UI */
    .stButton button {
        background-color: #40414F;
        color: white;
        border: 1px solid #565869;
    }
    
    /* Spinner personalizado */
    .stSpinner {
        border-color: #565869;
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    
    /* Contenedor principal */
    .main-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    
    /* Área de mensajes con scroll */
    .messages-container {
        flex-grow: 1;
        overflow-y: auto;
        padding: 2rem 0;
    }
    
    /* Input fijo en la parte inferior */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 800px;
        background-color: #343541;
        padding: 1rem;
        border-top: 1px solid #565869;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = None
    if "processed_url" not in st.session_state:
        st.session_state.processed_url = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

def process_input():
    if st.session_state.user_input:
        user_question = st.session_state.user_input
        st.session_state.user_input = ""
        
        with st.spinner("Procesando..."):
            messages = [
                {"role": "system", "content": st.session_state.system_prompt},
                *[{"role": "user" if i % 2 == 0 else "assistant", "content": msg} 
                  for i, msg in enumerate(sum(st.session_state.chat_history, ()))],
                {"role": "user", "content": user_question}
            ]
            
            response = st.session_state.llm.invoke(messages)
            st.session_state.chat_history.append((user_question, response.content))

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🌐 Chat con Website")
        url = st.text_input("URL del sitio web:")
        
        if url and url != st.session_state.processed_url:
            with st.spinner("Procesando el sitio web..."):
                try:
                    context = get_website_content(url)
                    st.session_state.llm, st.session_state.system_prompt = get_chat_model(context)
                    st.session_state.processed_url = url
                    st.session_state.chat_history = []
                    st.success("¡Sitio web procesado correctamente!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return
    
    # Main chat container
    with st.container():
        # Messages area
        with st.container():
            for question, answer in st.session_state.chat_history:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div>👤 <b>Tú:</b> {question}</div>
                </div>
                <div class="chat-message assistant-message">
                    <div>🤖 <b>Asistente:</b> {answer}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Input area
        if st.session_state.llm and st.session_state.system_prompt:
            st.text_input(
                "",
                placeholder="Envía un mensaje...",
                key="user_input",
                on_change=process_input,
                label_visibility="collapsed"
            )

if __name__ == "__main__":
    main() 


