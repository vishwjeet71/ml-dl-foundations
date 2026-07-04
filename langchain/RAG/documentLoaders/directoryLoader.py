from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path='books/',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load() # Load at once
docs = loader.lazy_load() # on demand loading

for doc in docs:
    print(doc.metadata)