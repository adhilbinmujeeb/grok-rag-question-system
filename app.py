import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from groq import Groq

# MongoDB Connection
client = MongoClient('mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['business_rag']
business_collection = db['business_attributes']
question_collection = db['questions']
listings_collection = db['business_listings']  # New collection for showcased listings

# Groq API Setup
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# Helper Functions
def get_business(business_name):
    return business_collection.find_one({"business_name": business_name})

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

def groq_qna(query, context=None):
    context_str = f"Context: {context}" if context else "No specific context provided."
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful business analyst. Use the context if provided."},
            {"role": "user", "content": f"{context_str}\n\nQuery: {query}"}
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
    "Machine Learning Clustering",
    "Company Valuation Estimator",  # New Feature
    "Interactive Business Assessment",  # New Feature
    "Showcase Listings for Investors"  # New Feature
])

# Session State
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'comments' not in st.session_state:
    st.session_state.comments = {}
if 'valuation_data' not in st.session_state:
    st.session_state.valuation_data = {}
if 'assessment_responses' not in st.session_state:
    st.session_state.assessment_responses = {}
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0

# Get list of business names
business_names = [b['business_name'] for b in get_all_businesses()]

# Existing Features (unchanged for brevity, only showing new ones below)
if page == "Business Performance Dashboard":
    st.header("Business Performance Dashboard")
    business_name = st.selectbox("Select Business", business_names)
    business = get_business(business_name)
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

elif page == "Smart Q&A":
    st.header("Smart Q&A (Powered by Groq)")
    query = st.text_input("Ask a question (e.g., 'How does this business make money?')")
    business_name = st.selectbox("Optional: Select Business for Context", ["None"] + business_names)
    if query and st.button("Submit"):
        if business_name != "None":
            business = get_business(business_name)
            response = groq_qna(query, str(business))
        else:
            response = groq_qna(query)
        st.write("**Response:**")
        st.markdown(response)

# ... (Other existing features remain unchanged for brevity)

# New Feature 1: Company Valuation Estimator
elif page == "Company Valuation Estimator":
    st.header("Company Valuation Estimator")
    st.write("Answer the questions to estimate your company's valuation.")

    # Questions to gather valuation data
    valuation_questions = [
        "What is your company's annual revenue (in USD)?",
        "What is your company's profitability status (e.g., Profitable, Break-even, Loss-making)?",
        "What is your company's growth rate (e.g., High, Moderate, Low)?",
        "What industry does your company operate in?"
    ]

    if 'valuation_step' not in st.session_state:
        st.session_state.valuation_step = 0

    if st.session_state.valuation_step < len(valuation_questions):
        current_question = valuation_questions[st.session_state.valuation_step]
        st.write(f"**Question {st.session_state.valuation_step + 1}:** {current_question}")
        answer = st.text_input("Your Answer", key=f"val_step_{st.session_state.valuation_step}")
        if st.button("Next"):
            st.session_state.valuation_data[current_question] = answer
            st.session_state.valuation_step += 1

    if st.session_state.valuation_step >= len(valuation_questions):
        st.write("**Collected Data:**", st.session_state.valuation_data)
        revenue = float(st.session_state.valuation_data.get(valuation_questions[0], "0").replace("$", "").replace(",", ""))
        profitability = st.session_state.valuation_data.get(valuation_questions[1], "Loss-making")
        growth = st.session_state.valuation_data.get(valuation_questions[2], "Low")
        industry = st.session_state.valuation_data.get(valuation_questions[3], "Other")

        # Simple valuation logic (e.g., revenue multiple)
        multiple = 2.0  # Default multiple
        if profitability == "Profitable": multiple += 1.0
        if growth == "High": multiple += 1.5
        if industry in ["Tech", "Healthcare"]: multiple += 0.5
        valuation = revenue * multiple

        st.write(f"**Estimated Valuation:** ${valuation:,.2f} (Based on a {multiple}x revenue multiple)")
        if st.button("Reset Valuation"):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}

# New Feature 2: Interactive Business Assessment
elif page == "Interactive Business Assessment":
    st.header("Interactive Business Assessment")
    st.write("Answer questions about your business. We'll adapt based on your responses.")

    questions = list(question_collection.find())
    if st.session_state.current_question_idx < len(questions):
        current_q = questions[st.session_state.current_question_idx]['question']
        st.write(f"**Question {st.session_state.current_question_idx + 1}:** {current_q}")
        response = st.text_input("Your Answer", key=f"q_{st.session_state.current_question_idx}")
        
        if st.button("Submit Answer"):
            st.session_state.assessment_responses[current_q] = response
            # Use Groq to generate a follow-up question
            follow_up = groq_qna(f"Given the answer '{response}' to '{current_q}', suggest a relevant follow-up question.")
            st.session_state.assessment_responses[follow_up] = None  # Placeholder for next answer
            st.session_state.current_question_idx += 1

    else:
        st.write("**Your Responses:**", st.session_state.assessment_responses)
        if st.button("Reset Assessment"):
            st.session_state.current_question_idx = 0
            st.session_state.assessment_responses = {}

# New Feature 3: Showcase Listings for Investors
elif page == "Showcase Listings for Investors":
    st.header("Showcase Listings for Investors")
    tab1, tab2 = st.tabs(["List Your Business", "Investor Dashboard"])

    with tab1:
        st.subheader("List Your Business")
        listing_name = st.text_input("Business Name")
        listing_industry = st.text_input("Industry")
        listing_revenue = st.number_input("Annual Revenue (USD)", min_value=0.0)
        listing_description = st.text_area("Business Description")
        listing_contact = st.text_input("Contact Info")
        
        if st.button("Submit Listing"):
            listing = {
                "business_name": listing_name,
                "industry": listing_industry,
                "revenue": listing_revenue,
                "description": listing_description,
                "contact": listing_contact,
                "listed_date": datetime.now().isoformat()
            }
            listings_collection.insert_one(listing)
            st.success("Business listed successfully!")

    with tab2:
        st.subheader("Investor Dashboard")
        listings = list(listings_collection.find())
        if listings:
            df = pd.DataFrame(listings)
            st.dataframe(df[['business_name', 'industry', 'revenue', 'description', 'contact']])
        else:
            st.write("No businesses listed yet.")

# Footer
st.sidebar.write(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
if __name__ == "__main__":
    st.write("Navigate using the sidebar!")
