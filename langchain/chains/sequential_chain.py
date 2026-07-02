from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

prompt1 = PromptTemplate(
    template= "write a detail report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "summarize this {text} in 2-3 lines",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "AI was born"})
print(result)

chain.get_graph().print_ascii()