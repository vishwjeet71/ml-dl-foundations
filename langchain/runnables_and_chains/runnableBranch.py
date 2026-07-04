from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id= "Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template= "Write a comprehensive report on the following topic: {topic}.",
    input_variables= ["topic"]
)

summary_prompt = PromptTemplate(
    template= "Write a concise summary of the following report: {report}.",
    input_variables= ["report"]
)
report_chain = RunnableSequence(prompt, model, parser)

condition_chain = RunnableBranch(
    (lambda x: len(x.split()) > 1000, RunnableSequence(summary_prompt, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, condition_chain)

print(final_chain.invoke({"topic":"global warming"}))