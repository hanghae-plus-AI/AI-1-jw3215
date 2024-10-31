import argparse
import sys
import os
sys.path.append(os.path.abspath(".."))

import sqlite3
from typing import List
from langchain.schema import Document
from src.data_loader import DataLoader
import bs4


DEFAULT_DB_NAME = "../documents.db"
# DB 초기화 및 테이블 생성
def initialize_db(db_name: str = DEFAULT_DB_NAME) -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 문서 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        metadata TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# 데이터 저장
def save_documents_to_db(documents: List[Document], db_name: str = DEFAULT_DB_NAME) -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Document 객체의 내용을 테이블에 삽입
    for doc in documents:
        cursor.execute('''
        INSERT INTO documents (content, metadata) VALUES (?, ?)
        ''', (doc.page_content, str(doc.metadata)))  # metadata를 문자열로 저장합니다.
    
    conn.commit()
    conn.close()

# 실행 부분
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서를 SQLite 데이터베이스에 저장합니다.")
    parser.add_argument('file_paths', nargs='+', help='처리할 파일들의 경로를 지정합니다.')
    parser.add_argument('--db_name', default=DEFAULT_DB_NAME, help='SQLite 데이터베이스 파일의 이름을 지정합니다.')

    args = parser.parse_args()
    
    # DB 초기화
    initialize_db(db_name=args.db_name)
    
    # DataLoader 사용
    data_loader = DataLoader(file_paths=args.file_paths)
    documents = data_loader.load_and_split()

    # 문서 저장
    save_documents_to_db(documents, db_name=args.db_name)

    # DB 초기화
    # initialize_db()
    # # 데이터 로딩
    # bs_kwargs = dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("editedContent")
    #     )
    # )
    
    # # DataLoader 사용 예제
    # web_paths = ["https://spartacodingclub.kr/blog/all-in-challenge_winner"]
    # data_loader = DataLoader(web_paths=web_paths, bs_kwargs=bs_kwargs)
    # documents = data_loader.load_and_split()

    # # 문서 저장
    # save_documents_to_db(documents)