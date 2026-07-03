from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )

parser = PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template="Classify the following review.\n\nReview:\n{review}\n\n{format_instructions}",
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive review:\n{review}",
    input_variables=["review"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative review:\n{review}",
    input_variables=["review"],
)
classifier_chain = prompt1 | model | parser

full_chain = RunnablePassthrough.assign(
    sentiment = classifier_chain | RunnableLambda(lambda r: r.sentiment)) | RunnableBranch(
    (lambda x: x["sentiment"] == "positive", prompt2 | model | StrOutputParser()),
    (lambda x: x["sentiment"] == "negative", prompt3 | model | StrOutputParser()),
    RunnableLambda(lambda _: "Could not determine sentiment")
)

print(full_chain.invoke({
    "review": "I am very disappointed with this phone. It overheats within minutes of charging, and it frequently has network issues. This is not the quality I expected. I would like either a full refund or a replacement."
}))

full_chain.get_graph().print_ascii()