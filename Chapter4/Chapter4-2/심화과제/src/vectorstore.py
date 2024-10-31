import sqlite3
import os
import json
from typing import List
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

DEFAULT_DB_NAME = "../documents.db"

class VectorStoreManager:
    def __init__(self, 
                 db_name: str = DEFAULT_DB_NAME, 
                 persist_directory: str = "vectorstore",
                 embedding_function=None):
        self.db_name = db_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function or OpenAIEmbeddings()

        self.vectorstore = self.create_vectorstore()

    def load_documents_from_db(self) -> List[Document]:
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT content, metadata FROM documents")
        rows = cursor.fetchall()
        conn.close()

        documents = []
        for content, metadata in rows:
            if metadata:
                metadata = metadata.replace("'", '"')
                metadata_dict = json.loads(metadata)
            else:
                metadata_dict = {}
            doc = Document(page_content=content, metadata=metadata_dict)
            documents.append(doc)
        return documents


    def create_vectorstore(self):
        documents = self.load_documents_from_db()
        # 벡터스토어 생성 (메모리에만 존재)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function
        )
        return vectorstore
    def load_or_create_vectorstore(self):
        pass

    def get_retriever(self):
        return self.vectorstore.as_retriever()

    def clear_cache(self):
        self.vectorstore._client.clear_system_cache()