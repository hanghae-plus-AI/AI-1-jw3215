from typing import List
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.schema import Document

class ChatBot:
    def __init__(self, retriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = hub.pull("rlm/rag-prompt")

    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_answer(self, user_msg: str) -> str:
        retrieved_docs = self.retriever.invoke(user_msg)
        print()
        print('#############################')
        print('#### RETRIEVED DOCUMENTS ####')
        print('#############################')
        print()
        print(retrieved_docs)

        user_prompt = self.prompt_template.invoke({
            "context": self.format_docs(retrieved_docs),
            "question": user_msg
        })
        response = self.llm.invoke(user_prompt)
        return str(response.content)