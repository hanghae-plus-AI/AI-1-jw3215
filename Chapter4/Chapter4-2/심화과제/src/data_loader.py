# from typing import Any, Dict, List, Optional
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema import Document

# class DataLoader:
#     def __init__(self, web_paths: List[str], bs_kwargs: Optional[Dict[str, Any]] = None):
#         self.web_paths = web_paths
#         self.bs_kwargs = bs_kwargs
#         self.loader = WebBaseLoader(
#             web_paths=self.web_paths,
#             bs_kwargs=self.bs_kwargs
#         )
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )

#     def load_and_split(self) -> List[Document]:
#         docs = self.loader.load()
#         splits = self.text_splitter.split_documents(docs)
#         return splits

from typing import List
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

class DataLoader:
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        self.loaders = []
        self._init_loaders()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    # 파일 경로에 따른 적절한 로더 초기화
    def _init_loaders(self):
        for path in self.file_paths:
            if path.lower().endswith('.pdf'):
                self.loaders.append(PyPDFLoader(path))
            elif path.lower().endswith('.txt'):
                self.loaders.append(TextLoader(path))
            else:
                print(f"지원하지 않는 파일 형식입니다: {path}")

    def load_and_split(self) -> List[Document]:
        docs = []
        for loader in self.loaders:
            documents = loader.load()
            docs.extend(documents)
        splits = self.text_splitter.split_documents(docs)
        return splits