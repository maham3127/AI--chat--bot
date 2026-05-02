import streamlit as st
import pickle
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# RESPONSES
# -----------------------------
responses = {
    "greeting": "Hello! Welcome to ARID University Helpdesk.",
    "goodbye": "Goodbye! Have a great day.",
    "admission": "Admissions are open for Sahiwal and Burewala campuses.",
    "programs": "We offer programs in CS, Health, Business, Arts, and Veterinary.",
    "cs_programs": "We offer BS CS, AI, Software Engineering, Web & Mobile Development.",
    "fees": "Fee details are available on the official website.",
    "location": "Sahiwal: Jail Road | Burewala: Chichawatni Road",
    "contact": "Helpline: 0335-111-8383 / 0334-111-8787",
    "health_programs": "Programs include Biochemistry, Microbiology, Nutrition.",
    "business_programs": "Programs include BBA, FinTech.",
    "arts_programs": "Programs include English, Fine Arts, Mathematics.",
    "veterinary_programs": "Livestock Assistant Diploma available."
}

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# UI SETUP
# -----------------------------
st.set_page_config(page_title="ARID University Helpdesk", page_icon="🎓")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Ask me about:")
    st.write("• Admissions")
    st.write("• Programs (CS, Health, Business, Arts, Veterinary)")
    st.write("• Fees")
    st.write("• Location & Contact")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# Main chat interface
st.title("🎓 ARID University Helpdesk")
st.caption("Your virtual assistant for university information")

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
if not st.session_state.chat:
    st.info("👋 Welcome! Type your question below to get started.")

for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin: 5px 0; color: black;'><b>👤 You:</b> {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0; color: black;'><b>🤖 Bot:</b> {msg['text']}</div>", unsafe_allow_html=True)

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.chat.append({"role": "user", "text": user_input})
    
    # Prediction
    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]
    confidence = np.max(model.predict_proba(vec))
    
    # Generate response
    if confidence < 0.4:
        response = "Sorry, I didn't understand. Please contact admin at 0335-111-8383"
    else:
        response = responses.get(intent, "I will get back to you shortly.")
    
    # Save bot message (ONCE)
    st.session_state.chat.append({"role": "bot", "text": response})
    
    # Rerun to display the new messages
    st.rerun()