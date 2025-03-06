import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from groq import Groq  # Import Groq API client

# MongoDB Connection
client = MongoClient('mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')  # Replace with your MongoDB URI
db = client['business_rag']
business_collection = db['business_attributes']
question_collection = db['questions']

# Groq API Setup
GROQ_API_KEY = "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"  # Replace with your actual Groq API key
groq_client = Groq(api_key=GROQ_API_KEY)

# Helper Functions
def get_business(business_id):
    return business_collection.find_one({"business_name": business_id})

def get_all_businesses():
    return list(business_collection.find())

def calculate_risk_score(business):
    risks = business.get('Business Attributes', {}).get('Risk Assessment', {})
    score = 0
    if risks.get('Operational Risks'): score += 20
    if risks.get('Market Risks'): score += 20
    if risks.get('Financial Risks'): score += 20
    return score

def get_performance_metrics(business):
    financials = business.get('Business Attributes', {}).get('Financial Metrics', {})
    growth = business.get('Business Attributes', {}).get('Growth & Scalability', {})
    return {
        'Revenue': financials.get('Revenue Brackets (Annual)', 'N/A'),
        'Profitability': financials.get('Profitability Status', 'N/A'),
        'Growth Rate': growth.get('Growth Rate', 'N/A')
    }

def match_question(query_embedding, questions):
    best_match = None
    highest_similarity = -1
    for q in questions:
        similarity = 1 - cosine(query_embedding, q['embedding'])
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = q
    return best_match

def calculate_exit_readiness(business):
    score = 0
    financials = business.get('Business Attributes', {}).get('Financial Metrics', {})
    growth = business.get('Business Attributes', {}).get('Growth & Scalability', {})
    market = business.get('Business Attributes', {}).get('Market Insights', {})
    if financials.get('Profitability Status') == 'Consistently Profitable': score += 30
    if growth.get('Scalability Potential') == 'Highly Scalable': score += 30
    if market.get('Market Position') == 'Market Leader': score += 20
    return score

def fetch_web_data(business_name):
    return f"Mock news for {business_name} fetched on {datetime.now().strftime('%Y-%m-%d')}"

def cluster_businesses(businesses):
    clusters = {}
    for b in businesses:
        industry = b.get('Business Attributes', {}).get('Business Fundamentals', {}).get('Industry Classification', {}).get('Primary Industry', 'Other')
        if industry not in clusters:
            clusters[industry] = []
        clusters[industry].append(b['business_name'])
    return clusters

# Groq-powered Q&A Function
def groq_qna(query, business_data=None):
    context = f"Business Data: {business_data}" if business_data else "General knowledge query."
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful business analyst. Use the provided context if available."},
            {"role": "user", "content": f"{context}\n\nQuery: {query}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# Streamlit App
st.title("Business Insights App with Groq Integration")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Business Performance Dashboard",
    "Smart Q&A",
    "Risk Assessment",
    "Competitive Analysis",
    "Growth Potential Predictor",
    "Business Profile Builder",
    "Watchlist & Alerts",
    "Community Insights",
    "Exit Readiness",
    "Acquirer Matchmaking",
    "Investment Opportunity Filter",
    "Trend Analysis",
    "External Data Integration",
    "Business Recommendations",
    "Machine Learning Clustering"
])

# Session State
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'comments' not in st.session_state:
    st.session_state.comments = {}

# 1. Business Performance Dashboard
if page == "Business Performance Dashboard":
    st.header("Business Performance Dashboard")
    business_id = st.selectbox("Select Business ID", [b['business_name'] for b in get_all_businesses()])
    business = get_business(business_id)
    if business:
        metrics = get_performance_metrics(business)
        st.subheader(f"{business['business_name']} Metrics")
        st.write(f"**Revenue:** {metrics['Revenue']}")
        st.write(f"**Profitability:** {metrics['Profitability']}")
        st.write(f"**Growth Rate:** {metrics['Growth Rate']}")
        fig, ax = plt.subplots()
        ax.bar(metrics.keys(), [1 if v == 'N/A' else 2 for v in metrics.values()], color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# 2. Smart Q&A System (Enhanced with Groq)
elif page == "Smart Q&A":
    st.header("Smart Q&A (Powered by Groq)")
    query = st.text_input("Ask a question (e.g., 'How does this business make money?')")
    business_id = st.selectbox("Optional: Select Business for Context", ["None"] + [b['business_name'] for b in get_all_businesses()])
    
    if query and st.button("Submit"):
        if business_id != "None":
            business = get_business(business_name)
            business_data = str(business)
            response = groq_qna(query, business_data)
        else:
            response = groq_qna(query)
        st.write("**Response:**")
        st.markdown(response)

# 3. Risk Assessment
elif page == "Risk Assessment":
    st.header("Risk Assessment")
    business_id = st.selectbox("Select Business ID", [b['business_name'] for b in get_all_businesses()], key="risk")
    business = get_business(business_id)
    if business:
        risk_score = calculate_risk_score(business)
        st.subheader(f"Risk Profile for {business['business_name']}")
        st.write(f"**Risk Score:** {risk_score}/60")
        st.progress(min(risk_score / 60, 1.0))

# 4. Competitive Analysis
elif page == "Competitive Analysis":
    st.header("Competitive Analysis")
    industry = st.selectbox("Select Industry", ["Tech", "Healthcare", "Finance", "Other"])
    businesses = business_collection.find({
        "Business Attributes.Business Fundamentals.Industry Classification.Primary Industry": industry
    })
    df = pd.DataFrame(list(businesses))
    if not df.empty:
        st.write(f"Businesses in {industry}:")
        st.dataframe(df[['business_name', 'Business Attributes.Financial Metrics.Revenue Brackets (Annual)']])
    else:
        st.write("No businesses found.")

# 5. Growth Potential Predictor
elif page == "Growth Potential Predictor":
    st.header("Growth Potential Predictor")
    business_id = st.selectbox("Select Business ID", [b['business_name'] for b in get_all_businesses()], key="growth")
    business = get_business(business_id)
    if business:
        growth = business.get('Business Attributes', {}).get('Growth & Scalability', {})
        score = 0
        if growth.get('Growth Rate') == 'High': score += 50
        if growth.get('Scalability Potential') == 'Highly Scalable': score += 50
        st.write(f"**Growth Potential Score for {business['business_name']}:** {score}/100")

# 6. Business Profile Builder
elif page == "Business Profile Builder":
    st.header("Build Your Business Profile")
    name = st.text_input("Business Name")
    industry = st.text_input("Primary Industry")
    revenue = st.selectbox("Revenue Bracket", ["< $1M", "$1M-$10M", "> $10M"])
    if st.button("Save Profile"):
        new_business = {
            "business_id": f"custom_{name.lower().replace(' ', '_')}",
            "business_name": name,
            "Business Attributes": {
                "Business Fundamentals": {"Industry Classification": {"Primary Industry": industry}},
                "Financial Metrics": {"Revenue Brackets (Annual)": revenue}
            }
        }
        business_collection.insert_one(new_business)
        st.success("Profile saved!")

# 7. Watchlist & Alerts
elif page == "Watchlist & Alerts":
    st.header("Watchlist & Alerts")
    business_id = st.selectbox("Add to Watchlist", [b['business_name'] for b in get_all_businesses()], key="watch")
    if st.button("Add to Watchlist"):
        if business_id not in st.session_state.watchlist:
            st.session_state.watchlist.append(business_id)
    st.write("**Your Watchlist:**", [get_business(bid)['business_name'] for bid in st.session_state.watchlist])
    if st.session_state.watchlist:
        st.write("Alert: Check your watchlist for updates!")

# 8. Community Insights
elif page == "Community Insights":
    st.header("Community Insights")
    business_id = st.selectbox("Select Business", [b['business_name'] for b in get_all_businesses()], key="community")
    comment = st.text_area("Add a Comment")
    if st.button("Submit Comment"):
        if business_id not in st.session_state.comments:
            st.session_state.comments[business_id] = []
        st.session_state.comments[business_id].append(comment)
        st.success("Comment added!")
    if business_id in st.session_state.comments:
        st.write("**Comments:**", st.session_state.comments[business_id])

# 9. Exit Readiness
elif page == "Exit Readiness":
    st.header("Exit Readiness")
    business_id = st.selectbox("Select Business ID", [b['business_name'] for b in get_all_businesses()], key="exit")
    business = get_business(business_id)
    if business:
        score = calculate_exit_readiness(business)
        st.write(f"**Exit Readiness Score for {business['business_name']}:** {score}/80")
        st.progress(min(score / 80, 1.0))

# 10. Acquirer Matchmaking
elif page == "Acquirer Matchmaking":
    st.header("Acquirer Matchmaking")
    business_id = st.selectbox("Select Business", [b['business_name'] for b in get_all_businesses()], key="acquirer")
    business = get_business(business_id)
    if business:
        industry = business.get('Business Attributes', {}).get('Business Fundamentals', {}).get('Industry Classification', {}).get('Primary Industry')
        matches = business_collection.find({
            "Business Attributes.Business Fundamentals.Industry Classification.Primary Industry": industry,
            "Business Attributes.Financial Metrics.Revenue Brackets (Annual)": {"$gt": business.get('Business Attributes', {}).get('Financial Metrics', {}).get('Revenue Brackets (Annual)', '')}
        })
        st.write("**Potential Acquirers:**")
        for m in matches:
            st.write(f"- {m['business_name']}")

# 11. Investment Opportunity Filter
elif page == "Investment Opportunity Filter":
    st.header("Investment Opportunity Filter")
    stage = st.selectbox("Development Stage", ["Pre-Revenue", "Early Stage", "Growth Stage"])
    businesses = business_collection.find({"Business Attributes.Business Fundamentals.Development Stage": stage})
    df = pd.DataFrame(list(businesses))
    if not df.empty:
        st.dataframe(df[['business_name', 'Business Attributes.Financial Metrics.Revenue Brackets (Annual)']])
    if st.button("Export to CSV"):
        df.to_csv("investment_opportunities.csv", index=False)
        st.success("Exported to investment_opportunities.csv")

# 12. Trend Analysis
elif page == "Trend Analysis":
    st.header("Trend Analysis")
    business_id = st.selectbox("Select Business", [b['business_name'] for b in get_all_businesses()], key="trend")
    mock_trends = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=5, freq="M"),
        "Revenue": [1e6, 1.2e6, 1.5e6, 1.8e6, 2e6]
    })
    st.line_chart(mock_trends.set_index("Date"))

# 13. External Data Integration
elif page == "External Data Integration":
    st.header("External Data Integration")
    business_id = st.selectbox("Select Business", [b['business_name'] for b in get_all_businesses()], key="external")
    business = get_business(business_id)
    if business:
        external_data = fetch_web_data(business['business_name'])
        st.write(f"**External Data:** {external_data}")

# 14. Business Recommendations (Enhanced with Groq)
elif page == "Business Recommendations":
    st.header("Business Recommendations (Powered by Groq)")
    interest = st.text_input("Enter your interests (e.g., 'tech startups')")
    if interest and st.button("Get Recommendations"):
        response = groq_qna(f"Recommend businesses based on interest: {interest}")
        st.write("**Recommendations:**")
        st.markdown(response)

# 15. Machine Learning Clustering
elif page == "Machine Learning Clustering":
    st.header("Machine Learning Clustering")
    businesses = get_all_businesses()
    clusters = cluster_businesses(businesses)
    for industry, names in clusters.items():
        st.write(f"**Cluster: {industry}**")
        st.write(", ".join(names))

# Footer
st.sidebar.write(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
if __name__ == "__main__":
    st.write("Navigate using the sidebar!")
