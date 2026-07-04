from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path= "https://github.com/vishwjeet71/VeritasAI"
)

page = loader.load()

print("Length of gitrepo:", len(page))
print("metadata:", page[0].metadata, end="\n\n")

# Chatbot

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
    template= "Analyze the given GitHub repository and explain its structure, purpose, and contents: {gitrepo}",
    input_variables= ["gitrepo"]
)

chain = prompt | model | StrOutputParser()

print(chain.invoke({"gitrepo": page[0].page_content}))