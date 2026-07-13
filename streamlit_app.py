import streamlit as st # web arayüzü (ui) oluşturmak için kullanacağımız kütüphane

from langchain_ollama import ChatOllama # ollama + gemma3
from langchain_huggingface import HuggingFaceEmbeddings # türkçe destekli embedding modelini LaBSE  yi buradan indir
from langchain_community.vectorstores import FAISS # vektör db
from langchain_community.document_loaders import PyPDFLoader # pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunk
from langchain_classic.chains import ConversationalRetrievalChain # chain = llm + memeory + vektor db
from langchain_classic.memory import ConversationBufferMemory # memory
import os
import tempfile # geçici dosya işlemleri için

# streamlit ile sayfa başlığı ve ikonu ayarla
st.set_page_config(
    page_title = "Müşteri Destek Botu",
    page_icon = "🤖"
)
st.title("Müşteri Destek Botu") # sayfa başlığı
st.write("Bir pdf yükleyin ve içeriğine dair sorular sorun. Türkçe desteklidir.")

@st.cache_resource(show_spinner="Embedding modeli yükleniyor, lütfen bekleyin...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

embedding = get_embedding_model()  # sunucu başlarken bir kez yükle

# pdf yükleme bileşeni
uploaded_file = st.file_uploader("PDF dosyanızı yükleyin", type="pdf", key="pdf_uploader")

# eğer kullanıcı yeni bir pdf yüklediyse ve daha önceki yüklenen ile aynı değilse
if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        # kullanıcıya işleniyor bilgisi gönder
        with st.spinner("PDF işleniyor, lütfen bekleyin..."):
            try:
                # yüklenen pdf i geçici bir dosyaya yazdır
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name # geçici dosyanın yolu

                # pdf i yükle
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                # metinleri parçala => chunklara böl
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(documents)

                # LaBSE embedding modeli (cache'den gelir, tekrar yüklenmez)

                # Faiss ile vektör veri tabanı oluşturma
                vektordb = FAISS.from_documents(docs, embedding)

                # memory ve gemma3:4b yi tanımla
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                llm = ChatOllama(model="gemma3:4b", temperature=0.2)

                # rag + memory chain oluştur
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vektordb.as_retriever(search_kwargs={"k": 3}),
                    memory=memory
                )
                st.session_state.qa_chain = qa_chain # chaini session state e kaydet
                st.session_state.chat_history = [] # konuşma geçmişi için boş bir liste oluştur
                st.session_state.last_uploaded_name = uploaded_file.name # son yüklenen dosya adını kaydet
                os.unlink(tmp_path) # geçici dosyayı sil
            except Exception as e:
                st.error(f"PDF işlenirken hata oluştu: {e}")
        st.success("PDF başarıyla işlendi!")

if "qa_chain" in st.session_state: # eğer pdf işlendiyse
    # kullanıcının sorusunu al
    user_question = st.text_input("Sorunuzu yazınız: ")

    if user_question:
        try:
            response = st.session_state.qa_chain.invoke(user_question)
            st.session_state.chat_history.append(("Siz", user_question))
            st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"Yanıt alınırken hata oluştu: {e}")

    if st.session_state.chat_history: # eğer konuşma geçmişi varsa
        st.subheader("Sohbet Geçmişi")
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}**: {msg}")
