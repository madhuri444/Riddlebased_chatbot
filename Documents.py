import os
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI


index_name = "chatbot"
namespace = "company-documents"
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def process_documents(directory_path="data"):
    document_data = []
    
    # Traverse through all files in the given directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            loader = PyPDFLoader(file_path)
            loaded_data = loader.load()

            # Extract document source and content
            document_source = loaded_data[0].metadata['source']
            document_content = loaded_data[0].page_content
            
            

            # Create file and folder details
            #file_name = document_source.split("/")[-1]
            #folder_names = document_source.split("/")[2:-1]
            file_name = document_source.split("\\")[-1]
            folder_names = document_source.split("\\")[1:-1]

            # Create a Document object
            doc = Document(
                page_content=f"<Source>\n{document_source}\n</Source>\n\n<Content>\n{document_content}\n</Content>",
                metadata={
                    "file_name": file_name,
                    "parent_folder": folder_names[1],
                    "folder_names": folder_names
                }
            )

            # Append the processed document to document_data
            document_data.append(doc)

    return document_data

# Call the function and store the result
#document_data = process_documents()



