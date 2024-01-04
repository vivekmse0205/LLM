import os
import traceback

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

import config
import utils

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY


class MarketingCompliance:
    def __init__(self):
        # Load the reference compliance data
        self.compliance_data = WebBaseLoader(config.COMPLIANCE_SITE).load()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        # Split the data into chunks
        self.compliance_chunks = self.text_splitter.transform_documents(self.compliance_data)
        self.store = LocalFileStore(config.STORAGE_PATH)
        # create an embedder
        core_embeddings_model = OpenAIEmbeddings()
        self.embedder = CacheBackedEmbeddings.from_bytes_store(
            core_embeddings_model,
            self.store,
            namespace=core_embeddings_model.model
        )
        # Create a vector db for storing embeddings of the contents of url
        self.vectorstore = FAISS.from_documents(self.compliance_chunks, self.embedder)
        self.retriever = self.vectorstore.as_retriever()
        # LLM model to use - can be modified to other models
        self.llm = OpenAIChat(temperature=0)
        self.handler = StdOutCallbackHandler()
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            callbacks=[self.handler],
            return_source_documents=True
        )
        self.prompt = PromptTemplate.from_template(config.PROMPT_TEMPLATE)

    def get_compliance_report(self, input_url):
        """

        :param input_url:
        :return:
        """
        try:
            input_loader = WebBaseLoader(input_url).load()
            rag_chain = (
                    {"context": self.retriever | utils.format_docs, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
            )
            result = rag_chain.invoke(f"{input_loader[0].page_content}")
            result = result.split('\n')
            return result

        except Exception as e:
            traceback.print_exc(e)
