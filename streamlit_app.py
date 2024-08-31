from dataclasses import dataclass
from typing import Literal, Any
import streamlit as st
import streamlit.components.v1 as components

from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: Any

def load_css():
    """Load custom CSS styles."""
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for conversation and history tracking."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = ChatGroq(
            api_key=st.secrets["llama3_api_key"],
            model="llama-3.1-70b-versatile",
            temperature=0
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm)
        )

def process_uploaded_file(uploaded_file):
    """Process the uploaded PDF file and create a retriever."""
    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.read())

    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    split_pages = splitter.split_documents(pages)
    
    vector_storage = FAISS.from_documents(split_pages, FakeEmbeddings(size=1352))
    return vector_storage.as_retriever()

def render_chat_interface():
    """Render the chat interface with the chat history."""
    chat_placeholder = st.container()
    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
<div class="chat-row 
    {'' if chat.origin == 'ai' else 'row-reverse'}">
    <img class="chat-icon" src="app/static/{
        'ai_icon.png' if chat.origin == 'ai' 
                      else 'user_icon.png'}"
         width=32 height=32>
    <div class="chat-bubble
    {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
            """
            st.markdown(div, unsafe_allow_html=True)
        
        for _ in range(3):
            st.markdown("")

# Define Question Template and Prompt
question_template = """
    You're a smart bot that answers questions based on the context given to you only.
    Return the answer.
    context: {context}
    question: {question}
"""

prompt = PromptTemplate.from_template(template=question_template)

def on_click_callback(retriever=None):
    """Handle the submit button click event for processing user input."""
    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt

        # If no retriever is provided, set a default context
        if retriever:
            result = RunnableParallel(context=retriever, question=RunnablePassthrough())
            ff = prompt.invoke(result.invoke(human_prompt))
        else:
            # Default response when no document is provided
            default_context = "There is no document uploaded. I can still answer general questions!"
            ff = prompt.invoke({"context": default_context, "question": human_prompt})
        
        ai_response = st.session_state.conversation.run(str(ff))
    
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", ai_response))
        st.session_state.token_count += cb.total_tokens

def setup_javascript():
    """Setup JavaScript to enable pressing Enter to submit the form."""
    components.html("""
    <script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Submit'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        switch (e.key) {
            case 'Enter':
                submitButton.click();
                break;
        }
    });
    </script>
    """, height=0, width=0)

# Main Application
load_css()
initialize_session_state()

st.title("Ask Chatbot 🤖")
st.header("Drop your document 💬")

# Initialize retriever as None
retriever = None

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:  # Check if a file is uploaded
    retriever = process_uploaded_file(uploaded_file)

render_chat_interface()

prompt_placeholder = st.form("chat-form")
with prompt_placeholder:
    st.markdown("** Ask questions about this document **")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="Hello bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=lambda: on_click_callback(retriever),
    )

credit_card_placeholder = st.empty()
credit_card_placeholder.caption(f"""
Used {st.session_state.token_count} tokens \n
Debug Langchain conversation: 
{st.session_state.conversation.memory.buffer}
""")

setup_javascript()
