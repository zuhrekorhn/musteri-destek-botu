"""
RAG pipeline bileşenlerini test eder.
Ollama/LLM gerektiren testler mock kullanır (CI ortamında LLM yok).
"""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@pytest.fixture(scope="session")
def embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="İade süresi 14 gündür. Ürünü kutusuyla iade edebilirsiniz."),
        Document(page_content="Şifrenizi 'Şifremi Unuttum' bağlantısından sıfırlayabilirsiniz."),
        Document(page_content="Yurt dışı teslimat henüz yapılmamaktadır."),
        Document(page_content="Faturanızı üye panelinden PDF olarak indirebilirsiniz."),
    ]


class TestChunking:
    def test_chunk_size_respected(self, sample_docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_documents(sample_docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= 120  # küçük tolerans

    def test_chunks_not_empty(self, sample_docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(sample_docs)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.page_content.strip() != ""


class TestVectorDB:
    def test_faiss_creation(self, embedding_model, sample_docs):
        db = FAISS.from_documents(sample_docs, embedding_model)
        assert db is not None

    def test_similarity_search_returns_results(self, embedding_model, sample_docs):
        db = FAISS.from_documents(sample_docs, embedding_model)
        results = db.similarity_search("iade nasıl yapılır", k=2)
        assert len(results) == 2

    def test_similarity_search_relevant(self, embedding_model, sample_docs):
        db = FAISS.from_documents(sample_docs, embedding_model)
        results = db.similarity_search("şifremi unuttum", k=1)
        assert "şifre" in results[0].page_content.lower() or "sıfırla" in results[0].page_content.lower()

    def test_faiss_save_and_load(self, embedding_model, sample_docs):
        db = FAISS.from_documents(sample_docs, embedding_model)
        with tempfile.TemporaryDirectory() as tmpdir:
            db.save_local(tmpdir)
            loaded_db = FAISS.load_local(tmpdir, embedding_model, allow_dangerous_deserialization=True)
            results = loaded_db.similarity_search("fatura", k=1)
            assert len(results) == 1


class TestExistingVectorStore:
    def test_vectorstore_files_exist(self):
        assert os.path.exists("faq_vectorstore/index.faiss"), "FAISS indeks dosyası eksik"
        assert os.path.exists("faq_vectorstore/index.pkl"), "FAISS pkl dosyası eksik"

    def test_vectorstore_loadable(self, embedding_model):
        db = FAISS.load_local(
            "faq_vectorstore",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        assert db is not None

    def test_vectorstore_query(self, embedding_model):
        db = FAISS.load_local(
            "faq_vectorstore",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        results = db.similarity_search("iade", k=3)
        assert len(results) > 0
