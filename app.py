import streamlit as st
import pandas as pd

st.title("OMEGA Energy Grid Dashboard")

df = pd.read_csv("merged_ontario_energy_dataset.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

st.line_chart(df["demand"])
