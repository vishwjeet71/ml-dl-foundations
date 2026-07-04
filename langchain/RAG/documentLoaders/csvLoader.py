from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Social_Network_Ads.csv")

docs = loader.load()

print("Length of document:", len(docs))
print("metadata:", docs[0].metadata)
print("content:", docs[0].page_content, end="\n\n")