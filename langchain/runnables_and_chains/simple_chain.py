from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

prompt = PromptTemplate(
    template= "What is the capital of {country}?",
    input_variables=["country"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"country": "india"})
print(result)