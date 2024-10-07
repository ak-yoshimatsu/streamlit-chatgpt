from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI

import streamlit as st


def init_page():
    st.set_page_config(page_title="My Greate ChatGPT", page_icon="🙇‍♂️")
    st.header("マイGPT")
    st.sidebar.title("オプション")


def init_message():
    clear_button = st.sidebar.button("履歴を削除する", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="何かお手伝いすることはありますか？")
        ]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("モデルを選択する: ", ("GPT-3.5", "GPT-4"))

    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    tempareture = st.sidebar.slider(
        "Temperature: ", min_value=0.0, max_value=2.0, value=0.0, step=0.01
    )

    return ChatOpenAI(temperature=tempareture, model_name=model_name)


def get_answer(llm: ChatOpenAI, messages: list):
    with get_openai_callback() as cb:
        answer = llm.invoke(messages)
    return answer.content, cb.total_cost


def main():
    init_page()

    llm = select_model()
    init_message()

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してな！"):
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # チャット履歴の表示
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")

    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == "__main__":
    main()
