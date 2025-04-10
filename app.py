import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

# -------- Password Protection --------
PASSWORD = st.secrets["general"]["password"]
password_input = st.text_input("Enter password", type="password")
if password_input != PASSWORD:
    st.warning("Please enter the correct password to access the dashboard.")
    st.stop()
st.success("Access granted!")

# -------- Load Iris Dataset --------
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['flower_name'] = df['target'].map(dict(zip(range(3), iris.target_names)))
    return df

df = load_data()

# -------- Sidebar Filters --------
st.sidebar.header("🔍 Filters")

# Filter by species
species_filter = st.sidebar.multiselect(
    "Species",
    options=df['flower_name'].unique(),
    default=df['flower_name'].unique()
)

# Contains filter on flower name
name_contains = st.sidebar.text_input("Flower name contains")

# Apply filters
filtered_df = df.copy()

if species_filter:
    filtered_df = filtered_df[filtered_df['flower_name'].isin(species_filter)]

if name_contains:
    filtered_df = filtered_df[filtered_df['flower_name'].str.contains(name_contains, case=False)]

st.write(f"🔎 Showing {len(filtered_df)} results after filtering.")

# -------- Compute Stats --------
def compute_stats(data: pd.DataFrame):
    return data.describe()

stats = compute_stats(filtered_df)

# -------- Display --------
st.subheader("📊 Aggregated Statistics")
st.dataframe(stats)

# Optional chart
st.subheader("📈 Sepal Width by Flower Name")
st.bar_chart(filtered_df.groupby('flower_name')['sepal width (cm)'].mean())

