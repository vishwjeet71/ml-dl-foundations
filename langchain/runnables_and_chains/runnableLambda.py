from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id= "Qwen/Qwen2.5-7B-Instruct",
        task= "text-generation",
        huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            
    )
)

prompt = PromptTemplate(
    template= "Write a poem based on the following topic: {topic}.",
    input_variables=["topic"]
)

def totalWords(text: str) -> int:
    return len(text.split())

lambda_function = RunnableLambda(totalWords)

chain = RunnablePassthrough.assign(
    poem = prompt | model | StrOutputParser()).assign(
    poem_length = lambda x: lambda_function.invoke(x["poem"])
)

print(chain.invoke({"topic": "rainy weather"}))