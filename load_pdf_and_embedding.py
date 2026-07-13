"""
1- kütüphanelri içeri aktar
2- SSS pdf oluştur ve yükle
3- chunkları oluştur
4- embedding uygula
5- vektör db oluştur
6- oluşturulan vektör db yi kaydet

"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# sık sorulan sorular dosyası yükle
loader = PyPDFLoader("musteri_destek_faq.pdf")
document = loader.load() # langchain document objesi oluştur
#print(document)

# metinleri parçalamak için
# splitter, metni anlamlı parçalara ayırma
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # her parça max 500 karakter içerecek
    chunk_overlap = 50 # her parça bir öncekinden 50 karakter alabilir
)

# chunkları oluştur
docs = text_splitter.split_documents(document)

#  türkçe için labse embedding yöntemi
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/LaBSE"
)

# parçalara ayrılmış metni embedding ile vekktör haline getir, indeks oluştur ve faiss de depola
vectordb = FAISS.from_documents(docs, embedding)

# oluşturulan vektör veri tabanını yerel diske kaydet
vectordb.save_local("faq_vectorstore")

print("Embedding ve vektör veri tabanı başarılı bir şekilde oluşturuldu.")
