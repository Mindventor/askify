import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from adjusted_ingest2 import create_vector_database_from_pdf, create_vector_database_from_url
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import pyttsx3
import speech_recognition as sr
import asyncio
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

ABS_PATH = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DB_DIR = os.path.join(ABS_PATH, "db")

# Initialize TTS engine
tts_engine = pyttsx3.init()

async def text_to_speech(text):
    """
    Convert text to speech in an async manner.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, tts_engine.say, text)
    await loop.run_in_executor(None, tts_engine.runAndWait)

def speech_to_text():
    """
    Convert speech to text using a microphone.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your prompt...")
        try:
            audio = recognizer.listen(source, timeout=20)
            st.info("Processing your input...")
            prompt_text = recognizer.recognize_google(audio)
            return prompt_text
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Speech Recognition service error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return None

def clear_database(db_dir):
    """
    Clear existing vector database only if it exists.
    """
    import shutil
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    os.makedirs(db_dir)

def initialize_qa_chain(vector_database):
    """
    Initialize the QA chain with the vector database and strict response guidelines.
    """
    try:
        llm = Ollama(
            model="mistral",
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        
        prompt_template = """Use only the following context to answer the question. If you cannot find a specific answer in the context, say "I cannot find an answer to this question in the provided content." Do not make up or infer information beyond what is explicitly stated in the context.

Context: {context}

Question: {question}

Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_database.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
            }
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None

def main():
    st.title('🦜🔗 ASKIFY')
    st.subheader('Get answers using text or voice, with text and speech responses')

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'vector_database' not in st.session_state:
        st.session_state.vector_database = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    source_type = st.radio("Choose your data source:", ["PDF", "Website URL"])

    if source_type == "PDF":
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])

        if uploaded_file is not None:
            temp_file_path = os.path.join(ABS_PATH, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    clear_database(DB_DIR)
                    try:
                        st.session_state.vector_database = create_vector_database_from_pdf(temp_file_path, DB_DIR)
                        if st.session_state.vector_database:
                            st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_database)
                            st.session_state.processed = True
                            st.success("PDF processed successfully! You can now ask questions below.")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")

    elif source_type == "Website URL":
        url = st.text_input("Enter the website URL:")

        if url and st.button("Process URL"):
            with st.spinner("Processing URL..."):
                clear_database(DB_DIR)
                try:
                    st.session_state.vector_database = create_vector_database_from_url(url, DB_DIR)
                    if st.session_state.vector_database:
                        st.session_state.qa_chain = initialize_qa_chain(st.session_state.vector_database)
                        st.session_state.processed = True
                        st.success("Website URL processed successfully! You can now ask questions below.")
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")

    # Chat interface with speech support
    if st.session_state.processed and st.session_state.vector_database and st.session_state.qa_chain:
        st.subheader("Chat with your data")
        
        chat_container = st.container()
        
        # Initialize user_query as None
        user_query = None
        
        # Add voice input option
        use_voice = st.checkbox("Use voice input")
        
        if use_voice:
            if st.button("Start Voice Input"):
                user_query = speech_to_text()
                if user_query:
                    st.write(f"Recognized text: {user_query}")
        else:
            user_query = st.text_input("Ask a question:", key="user_input")

        # Only process if user_query has a value
        if user_query:
            try:
                with st.spinner("Generating response..."):
                    response = st.session_state.qa_chain({"query": user_query})
                    answer = response['result']
                    sources = response.get('source_documents', [])
                    
                    st.session_state.chat_history.append({
                        "question": user_query,
                        "answer": answer,
                        "sources": sources
                    })
                    
                    # Add text-to-speech output
                    if st.checkbox("Read response aloud"):
                        asyncio.run(text_to_speech(answer))

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

        with chat_container:
            for chat in st.session_state.chat_history:
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                if chat.get('sources'):
                    st.markdown("**Source Segments:**")
                    for i, source in enumerate(chat['sources'], 1):
                        st.markdown(f"*Segment {i}:* {source.page_content[:200]}...")
                st.markdown("---")

if __name__ == '__main__':
    main()