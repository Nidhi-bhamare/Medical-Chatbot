import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Path to your FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"

# 🔧 Load the FAISS vector store with HuggingFace embeddings
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# 📌 Custom prompt template
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# 🔗 Load LLM from HuggingFace
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# 🌐 Main Streamlit app
def main():
    st.title("🩺 Ask Chatbot - Medical Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # 🧠 Custom system prompt
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, say "I don't know" and do not make up anything.
        Keep your answer focused and factual.

        Context: {context}
        Question: {question}

        Answer:
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.1"
        HF_TOKEN = # Replace with your real token

        try:
            # ✅ Load vectorstore
            vectorstore = get_vectorstore()
            print("✅ Vectorstore loaded")

            if vectorstore is None:
                st.error("❌ Failed to load the vector store.")
                return

            # ✅ Create the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            print("✅ QA chain created.")

            # 🔍 Ask the model
            response = qa_chain.invoke({'query': prompt})
            print("✅ Response received:", response)

            if "result" in response:
                result = response["result"]
                source_documents = response.get("source_documents", [])
                sources = "\n\n**Source Documents:**\n" + "\n".join(
                    [doc.metadata.get("source", "Unknown Source") for doc in source_documents]
                )
                result_to_show = result + sources

                st.chat_message("assistant").markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            else:
                st.warning("⚠️ Model did not return a result.")
                print("⚠️ Full response:", response)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            print("❌ Exception:", e)

if __name__ == "__main__":
    main()
