from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader

docs = """
Machine learning (ML) is a subset of artificial intelligence where 
algorithms learn to identify patterns in data and make predictions without
being explicitly programmed. Instead of relying on rigid,
human-written rules, ML models improve their accuracy over time
by analyzing large datasets.

The Core Approaches to Machine Learning
ML is generally divided into several key approaches based on how the computer
learns:

Supervised Learning: The model is trained on labeled data (where the correct
answer is already known) so it can predict outcomes for new, unseen data. Common 
tasks include classifying emails as spam or predicting housing
prices.

Unsupervised Learning: The algorithm analyzes unlabeled data to find hidden 
structures, clusters, or patterns on its own. This is often used for customer 
segmentation or anomaly detection.

Reinforcement Learning: The model learns by trial and error, receiving rewards or 
penalties for its actions to determine the best possible strategy. This method is
heavily used in robotics and game playing.

Generative AI: Powered by advanced deep learning, these models learn patterns
from existing data to create brand-new content like text, images, or audio.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 90,
    chunk_overlap=0
)

result = text_splitter.split_text(docs)

print("\n")
print("Normal text example:\n",result, end="\n\n")

# specific language Example

loader = TextLoader(
    file_path= 'codeExample.txt'
)

code = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

result = text_splitter.split_documents(code)
chunks = [c.page_content for c in result]
print("Code Example:\ntotal chunks of code:", len(chunks), end="\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk:{i+1}", chunk, end="\n")
    print("---")