import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
from groq import Groq

# Set page configuration
st.set_page_config(
    page_title="Business Insights Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        background-color: #E2E8F0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        padding-top: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #BFDBFE;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
    }
    .sidebar-header {
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# MongoDB Connection
try:
    client = MongoClient('mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    db = client['business_rag']
    business_collection = db['business_attributes']
    question_collection = db['questions']
    listings_collection = db['business_listings']
    print("Connected to MongoDB")  # Optional: Print a message to confirm connection
except pymongo.errors.ConnectionError as e:
    st.error(f"Failed to connect to MongoDB. Please check your connection details. Error: {e}")
    st.stop()

# Groq API Setup
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# Helper Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce DB calls
def get_business(business_name):
    return business_collection.find_one({"business_name": business_name})

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_all_businesses():
    return list(business_collection.find())

def match_question(query_embedding, questions):
    best_match = None
    highest_similarity = -1
    for q in questions:
        similarity = 1 - cosine(query_embedding, q['embedding'])
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = q
    return best_match

def groq_qna(query, context=None):
    try:
        context_str = f"Context: {context}" if context else "No specific context provided."
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert business analyst. Provide detailed, accurate, and actionable responses."},
                {"role": "user", "content": f"{context_str}\n\nQuery: {query}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with Groq API: {e}")
        return "Failed to get response from AI. Please try again later."


# Get list of business names
business_names = [b['business_name'] for b in get_all_businesses()]

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.title("💼 Business Insights Hub")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("", [
        "🔍 Smart Q&A",
        "💰 Company Valuation",
        "📊 Business Assessment",
        "🌐 Marketplace"
    ])

    st.markdown("---")
    st.markdown(f"<div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #64748B;'>{datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)

# Session State Initialization
if 'valuation_data' not in st.session_state:
    st.session_state.valuation_data = {}
if 'assessment_responses' not in st.session_state:
    st.session_state.assessment_responses = {}
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'valuation_step' not in st.session_state:
    st.session_state.valuation_step = 0
if 'sample_question' not in st.session_state:
    st.session_state.sample_question = None

# Pre-populate query from sample question if set
if st.session_state.sample_question:
    sample_query = st.session_state.sample_question
    st.session_state.sample_question = None  # Reset sample question
else:
    sample_query = ""


# 1. Smart Q&A
if "Smart Q&A" in page:
    st.markdown("# 🔍 Smart Business Intelligence")
    st.markdown("Get expert answers to your business questions powered by AI.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Ask a question about business strategy, valuation, market trends, etc.",
                              placeholder="E.g., How does this business make money?", value=sample_query) # Use sample query here
    with col2:
        business_name = st.selectbox("Select Business Context (Optional)", ["None"] + business_names)

    if query:
        submit_button = st.button("Get Insights", use_container_width=True)
        if submit_button:
            with st.spinner("Analyzing your question..."):
                if business_name != "None":
                    business = get_business(business_name)
                    response = groq_qna(query, str(business))
                else:
                    response = groq_qna(query)

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 💡 Expert Analysis")
                st.markdown(response)
                st.markdown("</div>", unsafe_allow_html=True)

    # Add sample questions for better user experience
    with st.expander("Sample Questions"):
        sample_questions = [
            "What are typical SaaS business valuation multiples?",
            "How can I improve my business's customer retention?",
            "What are common cash flow challenges for startups?",
            "How do I determine the right pricing strategy for my products?"
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.sample_question = q
                st.experimental_rerun()

# 2. Company Valuation Estimator
elif "Company Valuation" in page:
    st.markdown("# 💰 Company Valuation Estimator")
    st.markdown("Estimate your company's value using multiple industry-standard valuation methods.")

    # Progress bar
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

    total_steps = len(valuation_questions)
    current_step = st.session_state.valuation_step

    st.progress(current_step / total_steps)
    st.markdown(f"##### Step {current_step + 1} of {total_steps}")

    # Display the questions in a more attractive format
    if current_step < total_steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        current_question = valuation_questions[current_step]
        st.markdown(f"### {current_question}")

        # Add help text based on the question
        help_texts = {
            0: "Enter your total annual revenue before expenses.",
            1: "Enter your annual profit after all expenses and taxes.",
            2: "EBITDA = Earnings Before Interest, Taxes, Depreciation, and Amortization.",
            3: "Select the industry that best describes your business.",
            4: "Total value of all assets owned by your company.",
            5: "Total of all debts and obligations owed by your company.",
            6: "Estimate your cash flows for each of the next 5 years, separated by commas.",
            7: "Assess your company's growth trend compared to industry standards."
        }

        if current_step in help_texts:
            st.markdown(f"*{help_texts[current_step]}*")

        # Customized input methods based on question type
        if current_step == 0 or current_step == 1 or current_step == 2 or current_step == 4 or current_step == 5:
            answer = st.number_input("USD", min_value=0, step=1000, format="%i", key=f"val_step_{current_step}")
            answer = str(answer)
        elif current_step == 3:
            industries = ["Software/SaaS", "E-commerce", "Manufacturing", "Retail", "Healthcare", "Financial Services", "Real Estate", "Hospitality", "Technology", "Energy", "Other"]
            answer = st.selectbox("Select", industries, key=f"val_step_{current_step}")
        elif current_step == 6:
            year_cols = st.columns(5)
            cash_flows = []
            for i, col in enumerate(year_cols):
                with col:
                    cf = col.number_input(f"Year {i+1}", min_value=0, step=1000, format="%i", key=f"cf_{i}")
                    cash_flows.append(str(cf))
            answer = ",".join(cash_flows)
        elif current_step == 7:
            answer = st.select_slider("Select", options=["Low", "Moderate", "High"], key=f"val_step_{current_step}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if current_step > 0:
                if st.button("Back"):
                    st.session_state.valuation_step -= 1
                    st.experimental_rerun()

        with col2:
            if st.button("Next", use_container_width=True):
                st.session_state.valuation_data[current_question] = answer
                st.session_state.valuation_step += 1
                st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Show results after all questions are answered
    if current_step >= total_steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Company Information Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Industry:**")
            st.markdown("**Annual Revenue:**")
            st.markdown("**Net Income:**")
            st.markdown("**EBITDA:**")
        with col2:
            st.markdown(f"{st.session_state.valuation_data.get(valuation_questions[3], 'N/A')}")
            st.markdown(f"${float(st.session_state.valuation_data.get(valuation_questions[0], '0')):,.2f}")
            st.markdown(f"${float(st.session_state.valuation_data.get(valuation_questions[1], '0')):,.2f}")
            st.markdown(f"${float(st.session_state.valuation_data.get(valuation_questions[2], '0')):,.2f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Parse inputs for calculation
        revenue = float(st.session_state.valuation_data.get(valuation_questions[0], "0").replace("$", "").replace(",", "") or 0)
        earnings = float(st.session_state.valuation_data.get(valuation_questions[1], "0").replace("$", "").replace(",", "") or 0)
        ebitda = float(st.session_state.valuation_data.get(valuation_questions[2], "0").replace("$", "").replace(",", "") or 0)
        industry = st.session_state.valuation_data.get(valuation_questions[3], "Other")
        assets = float(st.session_state.valuation_data.get(valuation_questions[4], "0").replace("$", "").replace(",", "") or 0)
        liabilities = float(st.session_state.valuation_data.get(valuation_questions[5], "0").replace("$", "").replace(",", "") or 0)
        cash_flows_str = st.session_state.valuation_data.get(valuation_questions[6], "0,0,0,0,0")
        cash_flows = [float(cf.replace("$", "").replace(",", "")) for cf in cash_flows_str.split(",")]
        growth = st.session_state.valuation_data.get(valuation_questions[7], "Low")

        # Fetch industry benchmarks - Caching this would be beneficial if industry data is relatively static
        industry_data_cursor = business_collection.find({"Business Attributes.Business Fundamentals.Industry Classification.Primary Industry": industry})
        industry_data_list = list(industry_data_cursor) # Fetch all at once for mean calculation
        industry_avg_pe = 15.0  # Default P/E ratio
        industry_avg_ebitda_multiple = 8.0  # Default EV/EBITDA multiple
        if industry_data_list:
            pe_list = [b.get('Business Attributes', {}).get('Financial Metrics', {}).get('P/E Ratio', industry_avg_pe) for b in industry_data_list]
            ebitda_list = [b.get('Business Attributes', {}).get('Financial Metrics', {}).get('EV/EBITDA Multiple', industry_avg_ebitda_multiple) for b in industry_data_list]
            industry_avg_pe = np.mean([float(p) for p in pe_list if isinstance(p, (int, float)) and p > 0]) if any(isinstance(p, (int, float)) and p > 0 for p in pe_list) else industry_avg_pe
            industry_avg_ebitda_multiple = np.mean([float(e) for e in ebitda_list if isinstance(e, (int, float)) and e > 0]) if any(isinstance(e, (int, float)) and e > 0 for e in ebitda_list) else industry_avg_ebitda_multiple


        # Valuation calculation
        with st.spinner("Calculating company valuation..."):
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

            Format your response with clear headings and bullet points. Make sure to include a final summary section with a recommended valuation range at the end.
            """

            valuation_result = groq_qna(valuation_prompt)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Valuation Results")
        st.markdown(valuation_result)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Start New Valuation", use_container_width=True):
            st.session_state.valuation_step = 0
            st.session_state.valuation_data = {}
            st.experimental_rerun()

# 3. Interactive Business Assessment
elif "Business Assessment" in page:
    st.markdown("# 📊 Interactive Business Assessment")
    st.markdown("Get personalized insights through an adaptive business evaluation that responds to your inputs.")

    # Get questions from database - Caching this is good too if questions don't change frequently
    questions = list(question_collection.find())
    total_questions = len(questions)
    current_index = st.session_state.current_question_idx

    # Display progress
    if total_questions > 0:
        st.progress(min(1.0, current_index / total_questions))

    # Show current question or results
    if current_index < total_questions and questions:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        current_q = questions[current_index]['question']
        st.markdown(f"### Question {current_index + 1} of {total_questions}")
        st.markdown(f"**{current_q}**")

        response = st.text_area("Your Answer", height=100, key=f"q_{current_index}")

        if st.button("Submit Answer", use_container_width=True):
            st.session_state.assessment_responses[current_q] = response

            # Generate relevant follow-up with Groq
            with st.spinner("Analyzing your response..."):
                follow_up = groq_qna(f"Given the answer '{response}' to '{current_q}', suggest a relevant follow-up question that would help further assess this business area.")

            # Store the follow-up for later (consider if you want to use follow-up questions in the next steps, currently not used)
            # if follow_up and current_index + 1 >= total_questions:
            #     st.session_state.assessment_responses[follow_up] = None  # Storing follow-up but not using it

            st.session_state.current_question_idx += 1
            st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Show assessment summary and results
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Business Assessment Results")

        # Prepare data for analysis
        assessment_data = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.assessment_responses.items() if a is not None])

        # Generate analysis with Groq
        analysis_prompt = f"""
        You are an expert business consultant. Based on the following business assessment responses,
        provide a detailed analysis of the business strengths, weaknesses, and growth opportunities.

        Assessment Responses:
        {assessment_data}

        Please include:
        1. Executive Summary
        2. Key Strengths Identified
        3. Areas for Improvement
        4. Strategic Recommendations
        5. Next Steps

        Format your response with clear headings and bullet points.
        """

        with st.spinner("Generating business assessment report..."):
            analysis_result = groq_qna(analysis_prompt)

        st.markdown(analysis_result)

        if st.button("Start New Assessment", use_container_width=True):
            st.session_state.current_question_idx = 0
            st.session_state.assessment_responses = {}
            st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# 4. Showcase Listings for Investors
elif "Marketplace" in page:
    st.markdown("# 🌐 Business Marketplace")
    st.markdown("Connect businesses with investors. List your business or explore investment opportunities.")

    tabs = st.tabs(["🏢 List Your Business", "💸 Investor Dashboard"])

    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Create Your Business Listing")
        st.markdown("Complete the form below to showcase your business to potential investors.")

        col1, col2 = st.columns(2)
        with col1:
            listing_name = st.text_input("Business Name", placeholder="E.g., Acme Technologies")
            listing_industry = st.selectbox("Industry", [
                "Technology", "E-commerce", "Healthcare", "Finance",
                "Real Estate", "Manufacturing", "Retail", "Services",
                "Food & Beverage", "Education", "Other"
            ])
            listing_revenue = st.number_input("Annual Revenue (USD)", min_value=0, step=10000, format="%i")

        with col2:
            listing_location = st.text_input("Location", placeholder="City, Country")
            founding_year = st.number_input("Year Founded", min_value=1900, max_value=datetime.now().year, value=datetime.now().year) # Default to current year
            team_size = st.number_input("Team Size", min_value=1, value=5)

        listing_description = st.text_area("Business Description", height=150,
                                        placeholder="Describe your business, value proposition, market opportunity, and why investors should be interested.")

        col3, col4 = st.columns(2)
        with col3:
            investment_sought = st.number_input("Investment Amount Sought (USD)", min_value=0, step=50000, format="%i")
            equity_offered = st.slider("Equity Offered (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

        with col4:
            listing_contact = st.text_input("Contact Email", placeholder="your.email@example.com")
            website = st.text_input("Website URL", placeholder="https://yourbusiness.com")

        if st.button("Submit Listing", use_container_width=True):
            # Enhanced listing with more fields
            listing = {
                "business_name": listing_name,
                "industry": listing_industry,
                "revenue": listing_revenue,
                "description": listing_description,
                "contact": listing_contact,
                "location": listing_location,
                "founded": founding_year,
                "team_size": team_size,
                "investment_sought": investment_sought,
                "equity_offered": equity_offered,
                "website": website,
                "listed_date": datetime.now().isoformat()
            }
            listings_collection.insert_one(listing)
            st.success("✅ Business listed successfully! Your listing is now visible to investors.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Investor Dashboard")
        st.markdown("Explore businesses seeking investment. Filter by industry, investment size, and more.")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            industry_filter = st.multiselect("Filter by Industry", [
                "Technology", "E-commerce", "Healthcare", "Finance",
                "Real Estate", "Manufacturing", "Retail", "Services",
                "Food & Beverage", "Education", "Other"
            ])
        with col2:
            min_revenue = st.number_input("Minimum Revenue (USD)", min_value=0, step=50000, value=0)
        with col3:
            max_investment = st.number_input("Maximum Investment (USD)", min_value=0, step=100000, value=1000000)

        # Apply filters
        query = {}
        if industry_filter:
            query["industry"] = {"$in": industry_filter}
        if min_revenue > 0:
            query["revenue"] = {"$gte": min_revenue}
        if max_investment > 0:
            query["investment_sought"] = {"$lte": max_investment}

        # Get filtered listings
        listings = list(listings_collection.find(query))

        # Display listings in an attractive format
        if listings:
            for idx, listing in enumerate(listings):
                with st.container():
                    st.markdown(f"""
                    <div style='padding: 1.2rem; background-color: white; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='margin: 0; color: #1E3A8A;'>{listing.get('business_name', 'Unnamed Business')}</h3>
                            <span style='font-size: 0.8rem; background-color: #EFF6FF; padding: 0.2rem 0.5rem; border-radius: 4px; color: #1E3A8A;'>
                                {listing.get('industry', 'Uncategorized')}
                            </span>
                        </div>

                        <div style='display: flex; gap: 1rem; margin-top: 0.8rem; font-size: 0.85rem; color: #64748B;'>
                            <div><span style='font-weight: 500;'>📍 Location:</span> {listing.get('location', 'Not specified')}</div>
                            <div><span style='font-weight: 500;'>🏢 Founded:</span> {listing.get('founded', 'Not specified')}</div>
                            <div><span style='font-weight: 500;'>👥 Team:</span> {listing.get('team_size', 'Not specified')}</div>
                        </div>

                        <p style='margin-top: 0.8rem; margin-bottom: 0.8rem; font-size: 0.95rem;'>{listing.get('description', 'No description provided.')}</p>

                        <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;'>
                            <div>
                                <div style='font-weight: 500; color: #1E3A8A;'>Seeking ${listing.get('investment_sought', 0):,}</div>
                                <div style='font-size: 0.85rem; color: #64748B;'>For {listing.get('equity_offered', 0)}% equity</div>
                            </div>
                            <div>
                                <div style='font-weight: 500; color: #1E3A8A;'>Revenue: ${listing.get('revenue', 0):,}</div>
                                <div style='font-size: 0.85rem; color: #64748B;'>Annual</div>
                            </div>
                            <a href='mailto:{listing.get('contact', '')}' style='text-decoration: none; background-color: #1E3A8A; color: white; padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.9rem;'>Contact</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No businesses match your filter criteria. Try adjusting your filters or check back later for new listings.")

        st.markdown("</div>", unsafe_allow_html=True)

# Add a footer
st.markdown("""
<div style='background-color: #F8FAFC; padding: 1rem; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 2rem;'>
    Business Insights Hub © 2025 | Powered by Groq AI
</div>
""", unsafe_allow_html=True)
