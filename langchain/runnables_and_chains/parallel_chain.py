from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv
import os

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model2 = ChatHuggingFace(
    llm=llm
)

prompt1 = PromptTemplate(
    template="You have to generate a short and simple note for the given topic: {text}.",
    input_variables= ["text"]
)

prompt2 = PromptTemplate(
    template= "You have to generate 5 questions to test the user's understanding of this topic: {text}.",
    input_variables= ["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the given notes and questions.\n\nNotes:\n{notes}\n\nQuestions:\n{questions}",
    input_variables= ["notes", "questions"]
)

parser = StrOutputParser()

parrallel_chain = RunnableParallel(
    notes = prompt1 | model1 | parser,
    questions = prompt2 | model2 | parser
)

final_chain = parrallel_chain | prompt3 | model1 | parser

text = """
Linear regression is a foundational supervised machine learning technique that models the relationship between a dependent target variable and one or more independent predictor variables by fitting a straight line (or hyperplane) that minimizes the sum of squared errors.Key ConceptsFormula: The model follows \(y = mx + b\) for simple, or \(y = b_0 + b_1x_1 + \dots + b_nx_n\) for multiple variables.Methodology: Ordinary Least Squares (OLS) is commonly used to find the best-fit line by minimizing the sum of squared residuals.Evaluation: Performance is often measured using the \(R^{2}\) (R-squared) value, which indicates how well the model explains the variance in the data.Key AssumptionsFor accurate results, the data should exhibit linearity, independence of observations, homoscedasticity (constant variance of residuals), and minimal multicollinearity among predictors.ApplicationsLinear regression is frequently used for forecasting and trend analysis, commonly implemented in Python via scikit-learn or statsmodels.
"""

result = final_chain.invoke({"text": text})
print(result)

final_chain.get_graph().print_ascii()