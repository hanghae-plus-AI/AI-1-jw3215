from typing import Any, Dict, List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DataLoader:
    def __init__(self, web_paths: List[str], bs_kwargs: Optional[Dict[str, Any]] = None):
        self.web_paths = web_paths
        self.bs_kwargs = bs_kwargs
        self.loader = WebBaseLoader(
            web_paths=self.web_paths,
            bs_kwargs=self.bs_kwargs
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_and_split(self) -> List[Document]:
        docs = self.loader.load()
        splits = self.text_splitter.split_documents(docs)
        return splits