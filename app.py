# === ✅ MUST BE FIRST ===
import streamlit as st
st.set_page_config(page_title="Natural Language to SQL", layout="wide")

# === ✅ ENVIRONMENT SETUP ===
import os
from dotenv import load_dotenv
load_dotenv()

# === ✅ MODULE IMPORTS ===
from utils.llm import build_prompt, clean_sql_output, get_sql_from_llm
from utils.embeddings import load_vector_index, retrieve_relevant_schema
from utils.db import run_sql_query
from utils.plot import smart_plot
from utils.cost import estimate_cost

# === ✅ LOAD FAISS INDEX ===
index, schema_chunks = load_vector_index()

# === ✅ INIT SESSION STATE ===
for key in ["total_cost", "query_result", "query_sql", "cost_info", "schema_used"]:
    if key not in st.session_state:
        st.session_state[key] = 0.0 if key == "total_cost" else None

# === ✅ UI LAYOUT: MAIN (LEFT) + INFO (RIGHT) ===
left_col, right_col = st.columns([3, 1])

# === ✅ LEFT: INPUT & PROCESSING ===
with left_col:
    st.title("Ask Your Database Anything")
    query = st.text_area("What do you want to know?", height=100)

    if st.button("Run Query") and query:
        # Reset session
        st.session_state["query_result"] = None
        st.session_state["query_sql"] = None
        st.session_state["cost_info"] = None
        st.session_state["schema_used"] = None

        with st.spinner("Retrieving relevant schema..."):
            relevant_chunks = retrieve_relevant_schema(query, index, schema_chunks)
            st.session_state["schema_used"] = relevant_chunks

        with st.spinner("Generating SQL..."):
            prompt = build_prompt(query, relevant_chunks)
            raw_sql = get_sql_from_llm(prompt)
            cleaned_sql = clean_sql_output(raw_sql)
            st.session_state["query_sql"] = cleaned_sql
            st.code(cleaned_sql, language="sql")

        with st.spinner("Running SQL..."):
            df, error = run_sql_query(cleaned_sql)

            if df is not None:
                st.session_state["query_result"] = df
                st.success("Query executed successfully.")

                # Estimate and update cost
                cost_info = estimate_cost(query, relevant_chunks, cleaned_sql)
                st.session_state["cost_info"] = cost_info
                st.session_state["total_cost"] += cost_info["Total Cost"]
            else:
                st.error(f"SQL execution failed:\n\n{error}")

    if st.session_state["query_result"] is not None:
        st.subheader("Query Results")
        st.dataframe(st.session_state["query_result"])

        st.subheader("Visualization")
        smart_plot(st.session_state["query_result"])

# === ✅ RIGHT: COST + SCHEMA VIEWER ===
with right_col:
    st.markdown(
        f"#### 💸 Total Cost\n${st.session_state['total_cost']:.6f}",
        help="Estimated cumulative OpenAI API cost for this session"
    )

    if st.session_state["schema_used"]:
        with st.expander("Schema Chunks Used"):
            for chunk in st.session_state["schema_used"]:
                st.markdown(f"```sql\n{chunk}\n```")
