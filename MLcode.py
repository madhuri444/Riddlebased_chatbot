from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
#from google.colab import userdata
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import numpy as np
import tiktoken
import os
from groq import Groq
from dotenv import load_dotenv
import Documents


system_prompt = f"""You are an expert at understanding and analyzing company data - particularly shipping orders, purchase orders, invoices, and inventory reports.

#Answer any questions I have, based on the data provided. Always consider all of the context provided when forming a response.
"""

# Load environment variables from .env file
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

index_name = "chatbot"
namespace = "company-documents"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=index_name, embedding= embeddings)
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index(index_name)

vectorstore_from_documents = PineconeVectorStore.from_documents(
  [],
  embeddings,
  index_name=index_name,
  namespace=namespace
)
    


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def queryEmbedding(query):
    query_embedding_data= get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector= query_embedding_data.tolist(),top_k =10,include_metadata=True,namespace = namespace)
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    llm_response = groq_client.chat.completions.create(
    model = "llama-3.1-70b-versatile",
    messages = [

        {"role":"system","content":system_prompt},
        {"role":"user","content":augmented_query},
        ]
    )
    response = llm_response.choices[0].message.content
    
    return response
    



