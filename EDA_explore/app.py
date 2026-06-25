import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from eda_agent import DataManager, run_agent
from eda_agent.memory import list_sessions, load_session, save_session

st.set_page_config(layout="wide")

st.title("🧠 ChatGPT Data Analysis Agent (Local)")

# 初始化
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Session 選擇
sessions = list_sessions()
session_name = st.selectbox("選擇 Session", sessions + ["new"])

if session_name == "new":
    session_name = st.text_input("新 Session 名稱")

if session_name:
    st.session_state.messages = load_session(session_name)

# 上傳檔案
uploaded = st.file_uploader("上傳 CSV / Excel")

if uploaded:
    st.session_state.data_manager.load_file(uploaded)
    st.success("資料已載入")

    st.dataframe(st.session_state.data_manager.get_preview())

# 顯示歷史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 使用者輸入
if prompt := st.chat_input("請輸入分析問題"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    df = st.session_state.data_manager.df

    if df is None:
        response = "請先上傳資料"
        with st.chat_message("assistant"):
            st.write(response)
    else:
        code, result, error = run_agent(
            prompt,
            df,
            st.session_state.messages
        )

        with st.chat_message("assistant"):

            st.code(code, language="python")

            if error:
                st.error(error)
            else:
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)

                elif isinstance(result, pd.Series):
                    st.dataframe(result)

                elif result == plt:
                    st.pyplot(plt)

                else:
                    st.write(result)

        st.session_state.messages.append({
            "role": "assistant",
            "content": str(result)
        })

        save_session(session_name, st.session_state.messages)
