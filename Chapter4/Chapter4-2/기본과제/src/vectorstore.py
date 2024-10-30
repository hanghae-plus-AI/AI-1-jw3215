from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class VectorStoreManager:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=OpenAIEmbeddings()
        )

    def get_retriever(self):
        return self.vectorstore.as_retriever()
    
    def clear_cache(self):
        self.vectorstore._client.clear_system_cache()
