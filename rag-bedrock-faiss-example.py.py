# Standard libraries for handling JSON data, OS operations, and system functions
import json  # For handling JSON data
import os  # For interacting with the operating system
import sys  # For accessing system-specific parameters and functions

# AWS SDK for Python to interact with AWS services
import boto3  # For accessing AWS services, such as Bedrock

# Streamlit for building and deploying web applications
import streamlit as st  # For creating a web-based user interface

# Libraries from LangChain for working with language models and embeddings
from langchain_community.embeddings import BedrockEmbeddings  # For using AWS Bedrock embeddings
from langchain.llms.bedrock import Bedrock  # For accessing Bedrock language models

# NumPy for numerical operations
import numpy as np  # For numerical and array-based operations

# LangChain utilities for splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks

# LangChain community module for loading documents from PDFs
from langchain_community.document_loaders import PyPDFDirectoryLoader  # For loading PDF documents from a directory

# LangChain module for creating a vector store with FAISS
from langchain.vectorstores import FAISS  # For using FAISS as a vector store

# LangChain modules for handling prompts and question-answering chains
from langchain.prompts import PromptTemplate  # For creating custom prompt templates
from langchain.chains import RetrievalQA  # For creating a retrieval-based question-answering chain


# Initialize AWS Bedrock client
bedrock=boto3.client(service_name="bedrock-runtime")
# Set up Bedrock embeddings with the specified model
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

def data_ingestion():
    # Load documents from PDF files in the "data" directory
    loader=PyPDFDirectoryLoader("./data")
    documents=loader.load()

    # Split documents into chunks using a character-based splitter
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    # Create a FAISS vector store from documents using Bedrock embeddings
    vectorstore_faiss=FAISS.from_documents(docs, bedrock_embeddings)
    # Save the vector store locally
    vectorstore_faiss.save_local("./data/faiss_index")

def get_claude_llm():
    # Initialize Claude language model with AWS Bedrock
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock, model_kwargs={'maxTokens':512})
    return llm


def get_llama3_llm():
    # Initialize LLaMA3 language model with AWS Bedrock
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock, model_kwargs={'max_gen_len':512})
    return llm

# Define the prompt template for LLM responses
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

# Create a prompt template object
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm,vectorstore_faiss,query):
    # Create a retrieval-based QA chain using the specified LLM and vector store
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    # Get the answer from the QA chain
    answer=qa({"query":query})
    return answer['result']

def main():
    # Set up the Streamlit web app page configuration
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    # Get the user's question input
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        # Sidebar section to create or update the vector store
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                # Perform data ingestion and vector store creation
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            # Load vector store and use Claude LLM to get a response
            faiss_index = FAISS.load_local("./data/faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            # Load vector store and use LLaMA3 LLM to get a response
            faiss_index = FAISS.load_local("./data/faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    # Run the main function
    main()
