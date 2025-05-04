import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import easyocr

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# ------------------ App Config ------------------
st.set_page_config(page_title="FashionFit AI", layout="wide")
st.title("👗 FashionFit AI - Outfit Analyzer & Style Recommender")

# ------------------ File Upload ------------------
uploaded_image = st.file_uploader("📸 Upload Fashion Image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ------------------ Text Extraction with EasyOCR ------------------
    with st.spinner("🔍 Extracting text using EasyOCR..."):
        reader = easyocr.Reader(['en'])
        results = reader.readtext(np.array(img), detail=0)
        text = " ".join(results)

    st.text_area("🧾 Detected Text from Image:", text, height=150)

    # ------------------ Vector Store Creation ------------------
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    llm = Ollama(model="tinyllama")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # ------------------ Q&A Section ------------------
    st.subheader("🧠 Ask Fashion Questions")
    user_input = st.chat_input("What pairs well with this outfit?")
    if user_input:
        with st.spinner("💬 Generating response..."):
            answer = qa_chain.run(user_input)
            st.markdown(answer)

    # ------------------ Caption Generator ------------------
    st.divider()
    st.subheader("✍️ Generate Insta Blog Caption")
    if st.button("Create Fashion Caption"):
        prompt = "Write a stylish 100-word Instagram post about this outfit."
        with st.spinner("✨ Crafting your caption..."):
            blog = qa_chain.run(prompt)
            st.text_area("Generated Post:", blog, height=150)

    # ------------------ Visual Keyword Chart ------------------
    st.divider()
    st.subheader("🎨 Visualize Keywords")
    words = [w for w in text.split() if len(w) > 4]
    freq = {w: words.count(w) for w in set(words)}
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    if top:
        labels, counts = zip(*top)
        fig, ax = plt.subplots()
        sns.barplot(x=list(counts), y=list(labels), ax=ax)
        st.pyplot(fig)

else:
    st.info("⬅️ Upload a fashion-related image to begin.")
