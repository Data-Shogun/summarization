import os
import nltk

# Set writable NLTK path
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)

# Apply fix globally
os.environ["NLTK_DATA"] = nltk_data_path
nltk.data.path = [nltk_data_path]

# Pre-download necessary NLTK models
nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_path, quiet=True)
nltk.download("averaged_perceptron_tagger_eng", download_dir=nltk_data_path, quiet=True)


import streamlit as st
import validators

from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain

st.set_page_config(page_title='LangChain: Summarize Text from YT or Website', page_icon='🦜')
st.title('🦜 LangChain: Summarize Text from YT or Website')
st.subheader('Summarize URL')

api_key = st.sidebar.text_input('Groq API Key', value='', type='password')

if api_key:
    try:
        llm = ChatGroq(groq_api_key=api_key, model='Gemma2-9b-It')
        st.success("Model initialized successfully.")
        # Now continue using `llm` as needed
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
else:
    st.warning("Please enter your Groq API Key to continue.")

prompt_template = """
    Summarize the content provided for you to at least 300 words.
    content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

input_url = st.text_input('Enter a Youtube URL or any  website URL', label_visibility='collapsed')

if st.button('Summarize'):

    if not api_key or not input_url:
        st.error('Please enter the required information!')
    elif not validators.url(input_url):
        st.error('Please enter a valid URL')
    else:
        
        if 'youtube.com' in input_url:
            loader = YoutubeLoader.from_youtube_url(input_url)
        else:
            loader = UnstructuredURLLoader(
                urls=[input_url], 
                ssl_verification=False,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
            )
        with st.spinner('Please wait ...'):
            try:
                docs = loader.load()
                summarize_chain = load_summarize_chain(llm, chain_type='stuff', verbose=True)
                response = summarize_chain.run(docs)
                st.success(response)
            except Exception as e:
                st.exception(e)
