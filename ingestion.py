import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")

    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    documents = loader.load()


    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.getenv("index_name"))
    
    print("Ingestion finished!")