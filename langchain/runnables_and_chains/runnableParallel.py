from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id= "Qwen/Qwen2.5-7B-Instruct",
        task= "text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
)

parser = StrOutputParser()

post_prompt = PromptTemplate(
    template="Write a social media post about the following topic: {topic}.",
    input_variables= ["topic"]
)

blog_prompt = PromptTemplate(
    template="Write a medium-length blog post about the following topic: {topic}",
    input_variables= ["topic"]
)


seq_chain = RunnableParallel(
    blog = RunnableSequence(blog_prompt, model, parser),
    post = RunnableSequence(post_prompt, model, parser)
)

print(seq_chain.invoke({"topic": "How Far Are We from AGI?!"}))

seq_chain.get_graph().print_ascii()