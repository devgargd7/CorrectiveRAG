import logging
import os

import agent
import document_loader
import vectorstore_retriever
import workflow
import yaml_reader

YAML_FILE_PATH = "application.yaml"
logger = logging.getLogger(__name__)


class App:

    def __init__(
        self,
        doc_loader=document_loader.DocumentLoaderSplitter,
        embedding_model=None,
    ):
        self.vector_store_retriever = None
        self.app = None
        yaml_config_reader = yaml_reader.YamlConfigReader(YAML_FILE_PATH)
        self.collection_name = yaml_config_reader.get("collection_name")
        self.embedding_model = (
            embedding_model
            if embedding_model
            else yaml_config_reader.get("embedding_model")
        )
        self.model_name = yaml_config_reader.get("model")
        os.environ["GOOGLE_CSE_ID"] = yaml_config_reader.get("GOOGLE_CSE_ID")
        os.environ["GOOGLE_API_KEY"] = yaml_config_reader.get("GOOGLE_API_KEY")
        self.agent_helper = agent.AgentHelper(self.model_name)
        self.loader = doc_loader

    def ingest(self, pdf_file_paths=[], urls=[]):
        doc_loader_splitter = self.loader(urls, pdf_file_paths)
        doc_splits = doc_loader_splitter.load_and_split_documents()
        self.vector_store_retriever = vectorstore_retriever.VectorStoreRetriever(
            documents=doc_splits,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        retriever = self.vector_store_retriever.get_retriever()

        (
            retrieval_grader,
            summary_chain,
            rag_chain,
            hallucination_grader,
            answer_grader,
            question_router,
            web_search_tool,
        ) = self.agent_helper.get_agents()
        wf = workflow.Workflow(
            retriever,
            doc_splits,
            summary_chain,
            rag_chain,
            retrieval_grader,
            hallucination_grader,
            answer_grader,
            question_router,
            web_search_tool,
        )
        self.app = wf.build()

    def invoke(self, query, retries=2):
        if not self.app:
            return "Please add a PDF document or enter an URL."
        return self.app.stream({"question": query, "retries": retries})

    def clear(self):
        self.vector_store_retriever = None
        self.app = None
