from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="/home/vishwjeet/Desktop/ml-dl-foundations/langchain/RAG/documentLoaders/books/book-1.pdf")

docs = loader.load()

print("Length of document:", len(docs))
print("metadata:", docs[7].metadata)
print("content:", docs[7].page_content[:500], end="\n\n")

# with Chatbot

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id= 'Qwen/Qwen2.5-7B-Instruct',
        task= "text-generation",
        huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
)

prompt = PromptTemplate(
    template= "Summarize the given pdf Page in a very short summary: {document}",
    input_variables= ["document"]
)

chain = prompt | model | StrOutputParser()

print(chain.invoke({"document": docs[7].page_content}))