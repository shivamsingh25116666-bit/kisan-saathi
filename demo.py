import streamlit as st
import requests

st.set_page_config(page_title="Kisan-Agent Demo", page_icon="🌾")
st.title("🌾 Kisan-Agent (WhatsApp Simulator)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Send message as farmer (e.g. Ram)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hits your running FastAPI webhook endpoint directly
    try:
        res = requests.post(
            "http://localhost:8000/webhook/twilio", 
            data={"Body": prompt, "From": "whatsapp:+919876543210"}
        )
        
        # Grab the text returned by FastAPI and clean up XML tags if present
        if res.status_code == 200:
            raw_text = res.text
            if "<Message>" in raw_text and "</Message>" in raw_text:
                bot_reply = raw_text.split("<Message>")[1].split("</Message>")[0]
            else:
                bot_reply = raw_text if raw_text.strip() else "✅ Agent processed your request successfully!"
        else:
            bot_reply = f"⚠️ Error from backend (Status {res.status_code})"
            
    except Exception as e:
        bot_reply = f"❌ Could not connect to FastAPI backend: {e}"

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)