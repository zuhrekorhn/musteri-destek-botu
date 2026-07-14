import streamlit as st  # web arayüzü (ui) oluşturmak için kullanacağımız kütüphane

from langchain_ollama import ChatOllama  # ollama + gemma3
from langchain_huggingface import HuggingFaceEmbeddings  # türkçe destekli embedding modeli LaBSE
from langchain_community.vectorstores import FAISS  # vektör db
from langchain_community.document_loaders import PyPDFLoader  # pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # chunk
from langchain_classic.chains import ConversationalRetrievalChain  # chain = llm + memory + vektör db
from langchain_classic.memory import ConversationBufferMemory  # memory
import os
import tempfile  # geçici dosya işlemleri için

# ---------------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Müşteri Destek Botu",
    page_icon="🤖",
    layout="centered",
)

# Küçük görsel iyileştirmeler
st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; }
        .stChatMessage { border-radius: 14px; }
        div[data-testid="stFileUploader"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Kenar çubuğu — proje tanıtımı (LinkedIn/portföy için)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 Müşteri Destek Botu")
    st.caption("Türkçe destekli, belgeye dayalı RAG sistemi — tamamen local çalışır.")
    st.divider()
    st.markdown("**Nasıl çalışır?**")
    st.markdown(
        "1. PDF yükle\n"
        "2. Metin parçalara ayrılır (chunk)\n"
        "3. LaBSE ile vektöre çevrilir\n"
        "4. FAISS'te aranır\n"
        "5. Gemma3 Türkçe yanıt üretir"
    )
    st.divider()
    st.markdown("**Teknoloji**")
    st.markdown(
        "- LangChain — RAG\n"
        "- FAISS — vektör db\n"
        "- LaBSE — Türkçe embedding\n"
        "- Gemma3 (Ollama) — local LLM\n"
        "- Streamlit — arayüz"
    )
    st.divider()
    st.caption("github.com/zuhrekorhn/musteri-destek-botu")

# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.title("Müşteri Destek Botu 🤖")
st.write("Bir PDF yükleyin ve içeriğine dair sorular sorun. Türkçe desteklidir.")


@st.cache_resource(show_spinner="Embedding modeli yükleniyor, lütfen bekleyin...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")


embedding = get_embedding_model()  # sunucu başlarken bir kez yükle

# ---------------------------------------------------------------------------
# PDF yükleme ve chain kurulumu
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("PDF dosyanızı yükleyin", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    # Aynı dosya tekrar tekrar işlenmesin
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        with st.spinner("PDF işleniyor, lütfen bekleyin..."):
            try:
                # yüklenen pdf i geçici bir dosyaya yaz
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # pdf i yükle ve chunklara böl
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(documents)

                # FAISS vektör veri tabanı (embedding cache'den gelir)
                vektordb = FAISS.from_documents(docs, embedding)

                # memory + gemma3:4b + rag chain
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                llm = ChatOllama(model="gemma3:4b", temperature=0.2)
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vektordb.as_retriever(search_kwargs={"k": 3}),
                    memory=memory,
                )

                st.session_state.qa_chain = qa_chain
                st.session_state.chat_history = []
                st.session_state.last_uploaded_name = uploaded_file.name
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"PDF işlenirken hata oluştu: {e}")
        st.success("PDF başarıyla işlendi! Artık soru sorabilirsiniz. 👇")

# ---------------------------------------------------------------------------
# Sohbet arayüzü
# ---------------------------------------------------------------------------
if "qa_chain" in st.session_state:
    # Önceki mesajları baloncuklar halinde göster
    for role, msg in st.session_state.get("chat_history", []):
        with st.chat_message(role):
            st.markdown(msg)

    # Alt kısımda sabit soru kutusu
    if user_question := st.chat_input("Sorunuzu yazın... (örn: İade süresi kaç gün?)"):
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Yanıt üretiliyor..."):
                try:
                    response = st.session_state.qa_chain.invoke({"question": user_question})
                    answer = response["answer"]
                except Exception as e:
                    answer = f"⚠️ Yanıt alınırken hata oluştu: {e}"
            st.markdown(answer)

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer))
else:
    st.info("Başlamak için yukarıdan bir PDF yükleyin.")
