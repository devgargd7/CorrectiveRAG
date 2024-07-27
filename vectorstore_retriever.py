from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings


class VectorStoreRetriever:
    def __init__(self, documents, collection_name, embedding_model):

        self.documents = documents
        self.collection_name = collection_name
        self.embedding = NomicEmbeddings(model=embedding_model, inference_mode="local")
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )

    def get_retriever(self):
        return self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )
