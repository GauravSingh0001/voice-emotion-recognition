import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Analytics Dashboard")
st.markdown("### Usage statistics and insights")
st.markdown("---")

# Sample data (in real app, load from database)
emotions = ['Happy']*50 + ['Sad']*30 + ['Angry']*40 + ['Calm']*35 + ['Neutral']*25
confidence = [0.85]*50 + [0.78]*30 + [0.92]*40 + [0.88]*35 + [0.70]*25

df = pd.DataFrame({'emotion': emotions, 'confidence': confidence})

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions", len(df))
col2.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
col3.metric("Most Common", df['emotion'].mode()[0])
col4.metric("Accuracy", "88.19%")

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Emotion Distribution")
    counts = df['emotion'].value_counts()
    fig = px.pie(values=counts.values, names=counts.index)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Confidence Distribution")
    fig = px.histogram(df, x='confidence', nbins=20)
    st.plotly_chart(fig, use_container_width=True)
