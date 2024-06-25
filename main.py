from Workflow import Workflow
import os
from pprint import pprint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import VectorStoreRetriever, AgentHelper, YamlConfigReader, EnvironmentConfig, DocumentLoaderSplitter

if __name__ == "__main__":
    # Define parameters
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    yaml_file_path = "application.yaml"

    yaml_config_reader = YamlConfigReader.YamlConfigReader(yaml_file_path)

    collection_name = yaml_config_reader.get("collection_name")
    embedding_model = yaml_config_reader.get("embedding_model")
    model_name = yaml_config_reader.get("model")
    api_key = yaml_config_reader.get("api_key")

    os.environ["TAVILY_API_KEY"] = yaml_config_reader.get("TAVILY_API_KEY")

    # Instantiate and run workflow
    env_config = EnvironmentConfig.EnvironmentConfig(api_key=api_key)

    # Step 2: Load and Split Documents
    doc_loader_splitter = DocumentLoaderSplitter.DocumentLoaderSplitter(urls)
    doc_splits = doc_loader_splitter.load_and_split_documents()

    # Step 3: Create VectorStore and Retriever
    vector_store_retriever = VectorStoreRetriever.VectorStoreRetriever(documents=doc_splits, collection_name=collection_name, embedding_model=embedding_model)
    retriever = vector_store_retriever.get_retriever()

    agent_helper = AgentHelper.AgentHelper(yaml_config_reader)
    retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_router, web_search_tool = agent_helper.get_agents()
        
    workflow = Workflow(retriever, rag_chain, retrieval_grader, hallucination_grader, answer_grader, question_router, web_search_tool)
    app = workflow.build()
    
    # Test
    inputs = {"question": "What are the types of agent memory?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])

    # Compile
    # app = workflow.compile()
    inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])

    # question = "agent memory"
    # docs = retriever.invoke(question)
    # doc_txt = docs[1].page_content
    # result = grader.grade_retrieval(question, doc_txt)