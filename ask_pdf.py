from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def get_pdf_text(pdf_path):
    """
    Reads text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Combined text content from the PDF file.
    """
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        if page.extract_text():  # Handle cases where text extraction may fail
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits a large text into smaller chunks for processing.

    Args:
        text (str): The input text to split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """
    Creates a FAISS vector store for semantic search.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        FAISS: A FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    """
    Sets up a conversational retrieval chain using the LLaMA-3.2-3B-Instruct model.

    Args:
        vectorstore (FAISS): The vector store for document retrieval.

    Returns:
        ConversationalRetrievalChain: A LangChain conversation chain.
    """
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="float32"
    )

    # Create a Hugging Face pipeline
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Wrap pipeline in LangChain-compatible LLM
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Set up memory for conversational chain
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def main(pdf_path, question):
    """
    Main function to process a PDF and answer a question.

    Args:
        pdf_path (str): Path to the PDF file.
        question (str): The question to ask about the PDF content.
    """
    print("Reading PDF...")
    raw_text = get_pdf_text(pdf_path)

    print("Splitting text into chunks...")
    text_chunks = get_text_chunks(raw_text)

    print("Creating vector store...")
    vectorstore = get_vectorstore(text_chunks)

    print("Setting up conversation chain...")
    conversation_chain = get_conversation_chain(vectorstore)

    print("Generating response...")
    response = conversation_chain.run(question)
    
    # Check if response is a string or dictionary and print accordingly
    if isinstance(response, str):
        print("Response:")
        print(response)
    else:
        print("Response:")
        print(response.get('result', 'No result found'))  # In case it's a dictionary

if __name__ == "__main__":
    # Replace 'path_to_pdf.pdf' with the actual PDF file path
    pdf_path = "/content/The_Lightning_Thief_-_Percy_Jackson_1-10.pdf"
    
    # Replace 'your_question_here' with the actual question
    question = "What is the book about?"
    
    main(pdf_path, question)
