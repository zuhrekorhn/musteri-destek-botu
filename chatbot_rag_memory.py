"""
Problem tanımı: Akıllı müşteri destek sistemi: sık sorulan sorulara yanıt vereccek. Belgeye dayalı yanıt sistemi kuralım
    - Müşterielr sık sık benzer sorular sorarlar
        - şifremi unuttum, faturamı nereden alablirim, iade süresi kaç gün, yurt dışına gönderim yapıyormusunuz vb.
    - Çözüm: 
        - .pdf ( db, text, json...) dosyası formatında sıkça sorulan soruları vektör veri tabanına dönüştürürüz
        - kullanıcıdan gelen sorular sorgulanır ve dil modeli türkçe cevap üretir
Kullanılan teknolojiler:
    - Langchain: rag mimarisi kurmak için
    - faiss: embeddingleri saklamak için çnkü hızlı bir vektör veri tabanı
    - ollama: gemma3:1b soru cevap llm i için 
    - streamlit: web arayüzü, son  kullanıcı ile interaktif kullanıcı deneyimi

Veri Seti: geminiden faydalan
    - soru: yurt dışı satışlarınız bulunuyor mu? 
    - cevap: henüz bulunmuyor.

Plan/Program:
    - SSS içeren bir pdf oluştur
    - Kullanıcı bu dosyayı arayüzden yükleyecek
    - pdf metni chunklara ayrılır, embeddingler çıkarılır
    - kullanıcı soru sorduğu zaman vektör db den benzer içerikler getirilir, gemma ile cevap oluşturulur
    - memory(hafıza) ile konuşma geçmişi saklanır ve sonraki yanıtlara bağlam oluşturulur.

pip install langchain langchain-community sentence-transformers faiss-cpu pypdf ollama streamlit
"""

from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
import os

# embedding modelini başlat
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/LaBSE")

# daha önceden oluşturulmuş vektör veri tabanını yükle
vectordb = FAISS.load_local(
    "faq_vectorstore",
    embedding,
    allow_dangerous_deserialization = True
)

# konuşma geçmişi için memory oluştur
memory = ConversationBufferMemory(
    memory_key = "chat_history",
    return_messages = True
)

# llm tanımlama
llm = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.2
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectordb.as_retriever(search_kwargs = {"k":3}),
    memory = memory,
    verbose = True
)

# test
print("müşteri destek botuna hoşgeldinzi")
while True:
    user_input  = input("Siz: ")
    if user_input.lower() == "çık":
        break
    response = qa_chain.invoke({"question": user_input})
    print(f"Müşteri destek botu: {response['answer']}")