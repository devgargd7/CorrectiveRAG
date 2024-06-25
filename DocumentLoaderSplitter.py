from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentLoaderSplitter:
    def __init__(self, urls, chunk_size=250, chunk_overlap=0):
        self.urls = urls
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader = WebBaseLoader
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
    
    def load_and_split_documents(self):
        docs = [self.loader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]
        return self.text_splitter.split_documents(docs_list)