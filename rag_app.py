import os
from pprint import pprint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import vectorstore_retriever, agent, yaml_reader, document_loader, workflow

YAML_FILE_PATH = "application.yaml"

class App:

  def __init__(self):
    self.vector_store = None
    self.app = None
    yaml_config_reader = yaml_reader.YamlConfigReader(YAML_FILE_PATH)
    self.collection_name = yaml_config_reader.get("collection_name")
    self.embedding_model = yaml_config_reader.get("embedding_model")
    self.model_name = yaml_config_reader.get("model")
    os.environ["TAVILY_API_KEY"] = yaml_config_reader.get("TAVILY_API_KEY")
    self.agent_helper = agent.AgentHelper(self.model_name, yaml_config_reader)

  def ingest(self, pdf_file_paths=[], urls=[]):
    doc_loader_splitter = document_loader.DocumentLoaderSplitter(urls, pdf_file_paths)
    doc_splits = doc_loader_splitter.load_and_split_documents()

    self.vector_store_retriever = vectorstore_retriever.VectorStoreRetriever(
        documents=doc_splits, 
        collection_name=self.collection_name, 
        embedding_model=self.embedding_model
      )
    retriever = self.vector_store_retriever.get_retriever()

    retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_router, web_search_tool = self.agent_helper.get_agents()
    wf = workflow.Workflow(retriever, rag_chain, retrieval_grader, hallucination_grader, answer_grader, question_router, web_search_tool)
    self.app = wf.build()

  def invoke(self, query):
    if not self.app:
        return "Please, add a PDF document or enter an URL."
    return self.app.stream({"question": query})
  
  def clear(self):
    self.vector_store = None
    self.app = None