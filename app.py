import streamlit as st

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import textwrap
import os

os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    
    # Load Youtube Transcript
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])   
    
    chat = ChatOpenAI(temperature=0.2)
    
    # Template for system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

# --- Page Settings ---
page_icon = "🎬"
page_title = "VideoDigest"
layout = "centered"

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

# --- Hide Streamlit Style ---
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- Get Data From User --- 
with st.form("entry_form", clear_on_submit=True):
    video_url = st.text_input("Video URL:")
    submitted1 = st.form_submit_button("Search Video")

if submitted1:
    with st.form("entry_form", clear_on_submit=True):
    question = st.text_input("Question: ")
    submitted2 = st.form_submit_button("Ask Question")

if submitted2:
    db = create_db_from_youtube_video_url(video_url)
    query = question
    response, docs = get_response_from_query(db, query)
    st.write(textwrap.fill(response, width=50))
