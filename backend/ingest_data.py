import os 
from dotenv import load_dotenv 
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter 

def process_and_store_pdf():
    # Load environment variables
    load_dotenv() 

    # Get paths
    chroma_db_directory = os.getenv("CHROMA_DB", "db") 

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Handle PDF file path - using absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "PERSONAL PRODUCTIVITY PLAN.pdf")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        # Load and process PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        
        # Create vector database
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=chroma_db_directory
        )
        print("Document indexed successfully")
        return db

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

if __name__ == "__main__":
    process_and_store_pdf()