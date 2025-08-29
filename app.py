import streamlit as st
from vectordb import VectorDB
from llm_helper import get_words
import asyncio

st.set_page_config(page_title="Reverse Dictionary", page_icon="📚")
st.title("Reverse Dictionary")

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # no loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def format_results(results):
    for word, definition in zip(results["words"], results["definitions"]):
        with st.expander(f"{word}", expanded=False):
            st.write(definition)

user_input = st.text_input("Describe the word you're thinking of")
if st.button("Find word"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vector Search Based")
        db = VectorDB("reverse-dictionary")
        
        db_results = run_async(db.query_store(user_input))["matches"]
        db_results = {"words": [match["metadata"]["word"] for match in db_results],
                      "definitions": [match["metadata"]["description"] for match in db_results]}
        format_results(db_results)

    with col2:
        st.subheader("LLM Based")
        llm_results = get_words(user_input).parsed
        format_results(llm_results.dict())
