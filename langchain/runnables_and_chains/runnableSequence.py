from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(
    llm=llm
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="List 5 interesting facts about {topic}.",
    input_variables= ["topic"]
)

chain = RunnableSequence(prompt, model, parser)
print(chain.invoke({"topic": "AI"}))