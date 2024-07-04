from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata

class DocumentLoaderSplitter:
    def __init__(self, urls, pdf_file_paths, chunk_size=250, chunk_overlap=0):
        self.paths = urls if not pdf_file_paths else pdf_file_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader = WebBaseLoader if not pdf_file_paths else PyPDFLoader
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
    
    def load_and_split_documents(self):
        docs = [self.loader(path).load() for path in self.paths]
        docs_list = [item for sublist in docs for item in sublist]
        return filter_complex_metadata(self.text_splitter.split_documents(docs_list))