import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings # to get embedding model
from langchain_core.documents import Document # To store text and metadata
from langchain_text_splitters import CharacterTextSplitter # to split the large text into small chunks
from langchain_community.vectorstores import FAISS # to store the embedding for similarity search

key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key = key)

model = genai.GenerativeModel('gemini-2.5-flash')

def load_embedding():
    return HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

st.set_page_config('RAG ASSISTANT')
st.title('RAG Assistant :blue[Using Embedding and LLM]🎯')
st.subheader(':green[Your Intelligent Document Assistant!]📍')

with st.spinner('loading embedding model....👀'):
    embedding_model = load_embedding()
    
uploaded_file = st.file_uploader('Upload the document here in PDF format',type=['pdf'])

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    
    raw_text = ''
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    if raw_text.strip(): # ensures the pdf is not empty, removes spaces and check any content exist
        doc = Document(page_content = raw_text)
        splitter =CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        # max char in each chunk = 1000
        # overlap to maintain between context = 200
        
        chunk_text = splitter.split_documents([doc])
        # splits the document into multiple smaller chunks
        
        text = [ i.page_content for i in chunk_text]
        # converts the chunks into simple text in list
        
        vector_db = FAISS.from_texts(text, embedding_model)
        retrive = vector_db.as_retriever()
        
        st.success('Document processed and saved successfully!!! Ask your question')
        
        query = st.text_input('Enter your query here:')
        
        if query:
            with st.chat_message('human'):
                with st.spinner('Analyzing the document...'):
                    relevent_data = retrive.invoke(query) # this will find the most similar text chunks using FAISS
                    
                    content = '\n\n'.join([i.page_content for i in relevent_data]) # merges all relevant chunks into one text
                    
                    prompt = f'''You are an AI expert. Use the content generated to answer the query asked by the user. If you are unsure, you should say 'I am unsure about the question asked' Content: {content} Query: {query} Result: '''
                    
                    response = model.generate_content(prompt)
                    
                    st.markdown('### :green[Result]')
                    st.write(response.text)
                    
    else:
        st.warning('Drop the file in PDF format')