from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="book.pdf")
docs = loader.lazy_load()

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 90,
    chunk_overlap=10
)
result = text_splitter.split_documents(docs)

for data in result[:20]:
    print(data.page_content, end="\n")