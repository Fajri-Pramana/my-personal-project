
import streamlit as st
from bot import build_agent
import time # KRITIS: Memperbaiki NameError: name 'time' is not defined
import os 

st.set_page_config(page_title="Asisten Rekomendasi Kendaraan 🚗", layout="wide")

st.title("🤖 Asisten Rekomendasi Kendaraan")

if "agent" not in st.session_state:
    st.session_state.agent = build_agent() 

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Halo! Saya **Asisten Rekomendasi Kendaraan** Anda. Ada hal yang bisa saya bantu terkait pemilihan kendaraan yang anda butuhkan?",
    }]

agent = st.session_state.agent

reset_chat_button = st.button("🔄 Mulai Obrolan Baru")
if reset_chat_button:
    st.session_state.messages = []
    st.session_state.agent = build_agent() 
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Obrolan telah direset! Saya siap membantu Anda menemukan kendaraan impian. Apa preferensi Anda saat ini?",
    })


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)


user_input = st.chat_input("Apa kebutuhan kendaraan Anda?")

if user_input is not None:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Sedang berpikir..."):
        ai_output = ""
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for step in agent.stream({"input": user_input}):
                if "actions" in step.keys():
                    for action in step["actions"]:
                        tool_name = action.tool
                        tool_input = action.tool_input
                        tool_message = f"""
                            <div style="border-left: 5px solid #03A9F4; padding: 8px 12px; background-color: #e0f7fa; border-radius: 4px; font-size: 14px; margin-bottom: 5px;">
                                🛠️ Memanggil <b>{tool_name}</b> dengan input: <code>{tool_input}</code>
                            </div>
                        """
                        st.session_state.messages.append({"role": "assistant", "content": tool_message})
                        message_placeholder.markdown(tool_message, unsafe_allow_html=True)
                        time.sleep(0.1) 
                
                if "output" in step.keys():
                    full_response = step["output"]
            
            ai_output = full_response
            message_placeholder.markdown(ai_output, unsafe_allow_html=True)


    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_output,
    })
