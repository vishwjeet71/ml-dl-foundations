from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "This is a test document."
query_result = embedding.embed_query(text)
doc_result = embedding.embed_documents([text])

print(query_result)
print(doc_result)