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
listings_collection = db['business_listings']

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
            {"role": "system", "content": "You are an expert business valuation analyst. Provide detailed, accurate, and actionable responses."},
            {"role": "user", "content": f"{context_str}\n\nQuery: {query}"}
        ],
        max_tokens=1000  # Increased for detailed valuation responses
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
    "Company Valuation Estimator",
    "Interactive Business Assessment",
    "Showcase Listings for Investors"
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

# Existing Features (unchanged for brevity)
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

# Enhanced Feature: Company Valuation Estimator
elif page == "Company Valuation Estimator":
    st.header("Company Valuation Estimator")
    st.write("Provide details about your company to estimate its value using multiple valuation methods.")

    # Initial questions to gather data
    valuation_questions = [
        "What is your company's annual revenue (in USD)?",
        "What are your company's annual earnings (net income, in USD)?",
        "What is your company's EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization, in USD)?",
        "What industry does your company operate in?",
        "What is your company's total assets value (in USD)?",
        "What is your company's total liabilities (in USD)?",
        "What are your projected cash flows for the next 5 years (comma-separated, in USD)?",
        "What is your company's growth rate (e.g., High, Moderate, Low)?"
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
        
        # Parse inputs
        revenue = float(st.session_state.valuation_data.get(valuation_questions[0], "0").replace("$", "").replace(",", "") or 0)
        earnings = float(st.session_state.valuation_data.get(valuation_questions[1], "0").replace("$", "").replace(",", "") or 0)
        ebitda = float(st.session_state.valuation_data.get(valuation_questions[2], "0").replace("$", "").replace(",", "") or 0)
        industry = st.session_state.valuation_data.get(valuation_questions[3], "Other")
        assets = float(st.session_state.valuation_data.get(valuation_questions[4], "0").replace("$", "").replace(",", "") or 0)
        liabilities = float(st.session_state.valuation_data.get(valuation_questions[5], "0").replace("$", "").replace(",", "") or 0)
        cash_flows_str = st.session_state.valuation_data.get(valuation_questions[6], "0,0,0,0,0")
        cash_flows = [float(cf.replace("$", "").replace(",", "")) for cf in cash_flows_str.split(",")]
        growth = st.session_state.valuation_data.get(valuation_questions[7], "Low")

        # Fetch industry benchmarks from business_attributes
        industry_data = business_collection.find({"Business Attributes.Business Fundamentals.Industry Classification.Primary Industry": industry})
        industry_avg_pe = 15.0  # Default P/E ratio
        industry_avg_ebitda_multiple = 8.0  # Default EV/EBITDA multiple
        if industry_data:
            pe_list = [b.get('Business Attributes', {}).get('Financial Metrics', {}).get('P/E Ratio', industry_avg_pe) for b in industry_data]
            ebitda_list = [b.get('Business Attributes', {}).get('Financial Metrics', {}).get('EV/EBITDA Multiple', industry_avg_ebitda_multiple) for b in industry_data]
            industry_avg_pe = np.mean([float(p) for p in pe_list if isinstance(p, (int, float))]) if pe_list else industry_avg_pe
            industry_avg_ebitda_multiple = np.mean([float(e) for e in ebitda_list if isinstance(e, (int, float))]) if ebitda_list else industry_avg_ebitda_multiple

        # Groq Prompt for Valuation
        valuation_prompt = f"""
        You are an expert in business valuation. Given the following data about a company and industry benchmarks, calculate its valuation using all applicable methods:
        - Company Data:
          - Annual Revenue: ${revenue:,.2f}
          - Annual Earnings (Net Income): ${earnings:,.2f}
          - EBITDA: ${ebitda:,.2f}
          - Industry: {industry}
          - Total Assets: ${assets:,.2f}
          - Total Liabilities: ${liabilities:,.2f}
          - Projected Cash Flows (5 years): {', '.join([f'${cf:,.2f}' for cf in cash_flows])}
          - Growth Rate: {growth}
        - Industry Benchmarks:
          - Average P/E Ratio: {industry_avg_pe}
          - Average EV/EBITDA Multiple: {industry_avg_ebitda_multiple}

        Valuation Methods to Use:
        1. Market-Based:
           - Comparable Company Analysis (CCA): Use P/E Ratio (Company Value = Earnings × P/E Multiple) and EV/EBITDA.
           - Precedent Transactions: Suggest a multiplier based on industry norms if data is insufficient.
        2. Income-Based:
           - Discounted Cash Flow (DCF): Use a discount rate of 10% (WACC) unless industry suggests otherwise. Formula: Sum(CF_t / (1 + r)^t).
           - Earnings Multiplier (EV/EBITDA): Enterprise Value = EBITDA × Industry Multiple.
        3. Asset-Based:
           - Book Value: Assets - Liabilities.
           - Liquidation Value: Estimate based on assets (assume 70% recovery unless specified).

        Provide a detailed response with:
        - Calculated valuation for each method (if applicable).
        - Explanation of why each method is suitable or not for this company based on the industry and data.
        - A recommended valuation range combining the results.
        """
        
        with st.spinner("Calculating valuation with Groq..."):
            valuation_result = groq_qna(valuation_prompt)
        
        st.subheader("Valuation Results")
        st.markdown(valuation_result)
        
        if st.button("Reset Valuation"):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}

# Other Features (unchanged for brevity)
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
            follow_up = groq_qna(f"Given the answer '{response}' to '{current_q}', suggest a relevant follow-up question.")
            st.session_state.assessment_responses[follow_up] = None
            st.session_state.current_question_idx += 1
    else:
        st.write("**Your Responses:**", st.session_state.assessment_responses)
        if st.button("Reset Assessment"):
            st.session_state.current_question_idx = 0
            st.session_state.assessment_responses = {}

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
