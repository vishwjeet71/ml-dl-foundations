from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path="cricket.txt")

docs = loader.load()

print("Length of document:", len(docs))
print("metadata:", docs[0].metadata)
print("content:", docs[0].page_content[:500], end="\n\n")

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
    template= "Summarize the given document in a very short summary: {document}",
    input_variables= ["document"]
)

chain = prompt | model | StrOutputParser()

print(chain.invoke({"document": docs[0].page_content}))