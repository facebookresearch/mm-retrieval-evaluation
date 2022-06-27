from mmr.datasets import FireDataset
import streamlit as st
from pandasql import sqldf
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MMR_DATA", "data"))


@st.experimental_memo
def load_data():
    dataset = FireDataset.parse_file(DATA_DIR / "fire_aggregated_dataset.json")
    df = dataset.to_dataframe()
    df["n_labels"] = df["response_ids"].map(len)
    df["response_ids"] = df["response_ids"].map(lambda x: ",".join(map(str, x)))
    df["queue_ids"] = df["queue_ids"].map(lambda x: ",".join(map(str, x)))
    df["annotator_labels"] = df["annotator_labels"].map(lambda x: ",".join(map(str, x)))
    df["annotator_ids"] = df["annotator_ids"].map(lambda x: ",".join(map(str, x)))
    df["models"] = df["models"].map(lambda x: ",".join(map(str, x)))
    return df


@st.experimental_memo
def run_query(df, sql_query):
    return sqldf(sql_query, {"fire": df})


st.header("Query the FIRE Dataset with SQLite")
fire_df = load_data()
st.text(fire_df.columns.values)
query = st.text_area("SQL Query", "SELECT * FROM fire LIMIT 10")
run = st.button("Run Query")

if run:
    st.table(run_query(fire_df, query))
else:
    st.text("No queries run yet")
