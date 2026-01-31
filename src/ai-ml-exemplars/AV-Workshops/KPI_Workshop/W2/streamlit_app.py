import streamlit as st
from app_core import handle_question

st.title("KPI Copilot")
st.write("Ask KPI definition questions or request weekly metrics summaries.")

# Collect a user question and run the core pipeline.
q = st.text_input("Your question", value="Generate a weekly summary for 2025-W40 with conversion rate and AOV.")
if st.button("Run"):
    result = handle_question(q)
    if result["sql"]:
        st.subheader("SQL (read-only)")
        st.code(result["sql"], language="sql")
    st.subheader("Answer")
    st.write(result["response"])
