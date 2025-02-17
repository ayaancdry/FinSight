"""RAG (Retrieval-Augmented Generation) engine for document Q&A."""

from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from src.config import config


class RAGEngine:
    """
    RAG engine for context-aware question answering over documents.

    This class provides:
    - Document embedding using sentence-transformers
    - FAISS vector store for efficient similarity search
    - LLM-powered retrieval chain for Q&A
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the RAG engine.

        Args:
            openai_api_key: OpenAI API key. Uses config if not provided.
        """
        self.api_key = openai_api_key or config.openai_api_key
        self.embeddings = self._create_embeddings()
        self.vector_store: Optional[FAISS] = None
        self.retrieval_chain: Optional[RetrievalQA] = None

    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Create embeddings model for vectorizing text.

        Returns:
            HuggingFaceEmbeddings instance.
        """
        return HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def build_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Build FAISS vector store from documents.

        Args:
            documents: List of Document objects to index.

        Returns:
            FAISS vector store instance.
        """
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vector_store

    def _create_llm(self) -> ChatOpenAI:
        """
        Create LLM instance for generating responses.

        Returns:
            ChatOpenAI instance.
        """
        return ChatOpenAI(
            model=config.openai_model,
            api_key=self.api_key,
            temperature=0.1
        )

    def create_retrieval_chain(self) -> RetrievalQA:
        """
        Create the retrieval QA chain.

        Returns:
            RetrievalQA chain instance.

        Raises:
            ValueError: If vector store has not been built.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")

        llm = self._create_llm()
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retriever_k}
        )

        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        return self.retrieval_chain

    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.

        Args:
            question: User's question about the document.

        Returns:
            Dictionary containing 'result' and 'source_documents'.

        Raises:
            ValueError: If retrieval chain has not been created.
        """
        if self.retrieval_chain is None:
            raise ValueError("Retrieval chain not created. Call create_retrieval_chain first.")

        response = self.retrieval_chain.invoke({"query": question})
        return {
            "result": response.get("result", ""),
            "source_documents": response.get("source_documents", [])
        }

    def initialize(self, documents: List[Document]) -> None:
        """
        Complete initialization: build vector store and create chain.

        Args:
            documents: List of Document objects to index.
        """
        self.build_vector_store(documents)
        self.create_retrieval_chain()

    def get_relevant_chunks(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant document chunks without generating an answer.

        Args:
            query: Search query.
            k: Number of chunks to retrieve.

        Returns:
            List of relevant Document objects.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not built.")

        return self.vector_store.similarity_search(query, k=k)
