import os 
from dotenv import load_dotenv 
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def initialize_qa_chain():
    # Load environment variables
    load_dotenv()

    # Get API keys
    groq_api_key = os.getenv("GROQ_API_KEY") 
    if not groq_api_key: 
        raise ValueError("GROQ API key is not found in .env")

    chroma_db_directory = os.getenv("CHROMA_DB", "db")  

    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(
        persist_directory=chroma_db_directory,
        embedding_function=embeddings
    ) 
    
    # Advanced prompt template
    prompt_template = """
    You are a highly sophisticated Productivity Plan Assistant. Your responses should be:
    - Extremely well-structured
    - Visually appealing with Markdown formatting
    - Use tables for comparative data
    - Include bullet points for lists
    - Bold important concepts
    - Provide context before answering
    
    Context: {context}
    
    Question: {question}
    
    Guidelines for your response:
    1. Start with a brief summary if the question is complex
    2. Break down answers into logical sections
    3. Use this format for priorities/tasks:
       | Priority | Task | Status | Deadline |
       |----------|------|--------|----------|
    4. For progress reports, include:
       - Current status
       - Next steps
       - Potential blockers
    5. Always cite your sources
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize Groq chat model
    chat_model = ChatGroq(
        api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",  
        temperature=0.1,
        max_tokens=4000
    )

    # Create retrieval chain with custom prompt
    retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more context
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    ) 
    
    return qa_chain