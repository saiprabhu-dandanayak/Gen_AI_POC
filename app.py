import streamlit as st
import openai
import pandas as pd
import plotly.express as px
import random
import json
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(
    page_title="Next Best Action Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-top: 0;
    }
    .priority-high {
        color: #D32F2F;
        font-weight: bold;
    }
    .priority-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .priority-low {
        color: #4CAF50;
        font-weight: bold;
    }
    .action-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #1E88E5;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #D32F2F;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #757575;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = {}
if 'call_transcript' not in st.session_state:
    st.session_state.call_transcript = ""
if 'travel_notice_data' not in st.session_state:
    st.session_state.travel_notice_data = {}
if 'recent_transaction' not in st.session_state:
    st.session_state.recent_transaction = {}
if 'recommended_actions' not in st.session_state:
    st.session_state.recommended_actions = []
if 'sentiment_result' not in st.session_state:
    st.session_state.sentiment_result = {}
if 'chain_of_thought' not in st.session_state:
    st.session_state.chain_of_thought = ""
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Main title and description
st.markdown("<h1 class='main-header'>🎯 Next Best Action Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Customer Support Analysis & AI-Powered Recommendation System</p>", unsafe_allow_html=True)
st.markdown("---")

# OpenAI API key input in sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    if openai_api_key:
        openai.api_key = openai_api_key
        st.session_state.api_key_set = True
    
    model_option = st.selectbox(
        "Select AI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    )
    
    st.markdown("---")

# Function to analyze sentiment and generate recommendations using OpenAI
def analyze_with_ai(transcript, customer_data, travel_notice, transaction, model="gpt-3.5-turbo"):
    try:
        # Prompt for sentiment analysis
        sentiment_prompt = f"""
        Analyze the sentiment in this customer service transcript and return a JSON with:
        1. The sentiment (POSITIVE, NEGATIVE, or NEUTRAL)
        2. A confidence score between 0 and 1
        3. The key emotions detected
        4. A summary of customer frustration points

        Transcript:
        {transcript}

        Return only the JSON with no additional text.
        """
        
        sentiment_response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert customer service analyst."},
                {"role": "user", "content": sentiment_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        sentiment_result = json.loads(sentiment_response.choices[0].message.content)
        
        # Prompt for next best action recommendations
        action_prompt = f"""
        Based on the following information, recommend the next best actions for a bank agent to take:

        CUSTOMER INFORMATION:
        {json.dumps(customer_data, indent=2)}

        RECENT TRANSACTION:
        {json.dumps(transaction, indent=2)}

        TRAVEL NOTICE:
        {json.dumps(travel_notice, indent=2)}

        CALL TRANSCRIPT:
        {transcript}

        SENTIMENT ANALYSIS:
        {json.dumps(sentiment_result, indent=2)}

        Return a JSON array with the top 5 recommended actions, each containing:
        1. "action": a short action title
        2. "description": detailed description of what the agent should do
        3. "priority": "High", "Medium", or "Low"
        4. "icon": an appropriate emoji for the action
        5. "category": the category of action (e.g., "Technical Resolution", "Customer Service", "Sales Opportunity")
        
        Return only the JSON array with no additional text.
        """
        
        action_response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in banking customer service strategy."},
                {"role": "user", "content": action_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        recommended_actions = json.loads(action_response.choices[0].message.content)
        
        # Prompt for reasoning chain of thought
        reasoning_prompt = f"""
        Provide a detailed step-by-step chain of thought analysis for this customer service situation:

        CUSTOMER INFORMATION:
        {json.dumps(customer_data, indent=2)}

        RECENT TRANSACTION:
        {json.dumps(transaction, indent=2)}

        TRAVEL NOTICE:
        {json.dumps(travel_notice, indent=2)}

        CALL TRANSCRIPT:
        {transcript}

        SENTIMENT ANALYSIS:
        {json.dumps(sentiment_result, indent=2)}

        Include sections for:
        1. Customer Context Analysis
        2. Problem Identification
        3. Emotional Impact Assessment
        4. Priority Determination
        5. Opportunity Analysis
        6. Long-term Relationship Considerations

        Give a thorough, thoughtful analysis that demonstrates expert-level understanding of banking customer service.
        """
        
        reasoning_response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in customer psychology and banking operations."},
                {"role": "user", "content": reasoning_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        chain_of_thought = reasoning_response.choices[0].message.content
        
        return sentiment_result, recommended_actions, chain_of_thought
    
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return {
            "sentiment": "ERROR",
            "confidence": 0,
            "emotions": ["Error in analysis"],
            "frustration_points": ["API error occurred"]
        }, [], "Error generating analysis"

# Demo templates for quick testing
demo_templates = {
    "Card Declined While Traveling": {
        "customer_data": {
            "name": "Priya Sharma",
            "account_type": "Premium Checking",
            "account_opened": "March 12, 2018",
            "credit_score": 760,
            "average_balance": "$8,500",
            "card_type": "World Traveler Visa",
            "eligible_for_upgrade": True,
            "contact_preference": "Mobile",
            "phone": "+1 (555) 123-4567",
            "email": "priya.sharma@email.com"
        },
        "call_transcript": """
        Agent: Thank you for calling International Bank. This is Alex speaking. How may I assist you today?

        Priya: Hi, I'm extremely frustrated right now. I'm in Paris, and my card was just declined at a restaurant, which was very embarrassing. I specifically submitted a travel notice before leaving!

        Agent: I'm sorry to hear that, Ms. Sharma. Could you please confirm your full name and the last four digits of your account for verification?

        Priya: Priya Sharma, and the last four digits are 7842.

        Agent: Thank you for that information. I see you're calling about a declined transaction. Let me check what's happening with your account.

        Priya: Yes, please do. I submitted a travel notice last week specifically to avoid this situation! I'm only here for three more days and need my card to work.

        Agent: I understand your frustration. Let me look into your travel notice... I can see that you submitted a travel notice, but it seems there might be an issue with the dates or countries listed. Can you confirm when you submitted it and which countries you included?

        Priya: I submitted it on May 2nd, and I listed France, Italy, and Spain for May 5th through May 15th. I'm in Paris right now, so France should definitely be covered!

        Agent: I see the issue now. It appears your travel notice was processed, but there was a system error that prevented it from being properly activated. I sincerely apologize for this inconvenience.

        Priya: That's unacceptable! I rely on this card when traveling, and now I'm stuck without access to my funds in a foreign country!

        Agent: You're absolutely right, and I apologize again for this situation. I'll correct this immediately.
        """,
        "travel_notice_data": {
            "submitted_date": "May 2, 2023",
            "travel_start": "May 5, 2023",
            "travel_end": "May 15, 2023",
            "countries": ["France", "Italy", "Spain"],
            "status": "Submitted but not activated due to system error",
            "submission_channel": "Mobile App"
        },
        "recent_transaction": {
            "date": "May 7, 2023",
            "merchant": "Le Bistro Parisien",
            "location": "Paris, France",
            "amount": "€78.50",
            "status": "Declined",
            "reason": "Travel notice not active",
            "card_used": "World Traveler Visa ending in 7842"
        }
    },
    "Unauthorized Charges Dispute": {
        "customer_data": {
            "name": "Michael Chen",
            "account_type": "Everyday Checking",
            "account_opened": "August 23, 2020",
            "credit_score": 715,
            "average_balance": "$3,200",
            "card_type": "Cashback Rewards Mastercard",
            "eligible_for_upgrade": False,
            "contact_preference": "Email",
            "phone": "+1 (555) 987-6543",
            "email": "michael.chen@email.com"
        },
        "call_transcript": """
        Agent: Thank you for calling International Bank. This is Jamie speaking. How may I assist you today?

        Michael: Hi Jamie, I'm calling because I found some charges on my account that I definitely didn't make. I think my card information might have been stolen.

        Agent: I'm sorry to hear that, Mr. Chen. I'd be happy to help you with this situation. Could you please confirm your full name and the last four digits of your account for verification?

        Michael: Michael Chen, and the last four digits are 3456.

        Agent: Thank you, Mr. Chen. Can you tell me which charges you believe are unauthorized?

        Michael: Yes, there are three charges from yesterday. One is from "TechGadgetStore" for $429.99, another from "GameStreamPlus" for $59.99, and a third from "QuickFoodDelivery" for $42.50. I didn't make any of these purchases.

        Agent: I understand. Let me pull up your recent transaction history... Yes, I can see those three transactions from yesterday. Thank you for reporting this promptly.

        Michael: Will I be able to get my money back? And how did this happen? I'm always careful with my card.

        Agent: We take fraud very seriously, and yes, we'll begin the dispute process immediately. Even with careful handling, card information can sometimes be compromised through data breaches or skimming devices. Have you noticed anything unusual when using your card recently?

        Michael: Not really, but I did use it at a gas station I don't normally go to about a week ago. The card reader seemed a bit loose, but I didn't think much of it at the time.

        Agent: That could potentially be the source. Some criminals install skimming devices on card readers. I'd like to go ahead and cancel your current card immediately and issue you a new one with different numbers.
        """,
        "travel_notice_data": {
            "submitted_date": "N/A",
            "travel_start": "N/A",
            "travel_end": "N/A",
            "countries": [],
            "status": "No active travel notices",
            "submission_channel": "N/A"
        },
        "recent_transaction": {
            "date": "June 12, 2023",
            "merchant": "TechGadgetStore",
            "location": "Online",
            "amount": "$429.99",
            "status": "Completed",
            "reason": "N/A",
            "card_used": "Cashback Rewards Mastercard ending in 3456"
        }
    }
}

# Tabs for Input and Results
input_tab, results_tab = st.tabs(["📝 Input Data", "📊 Analysis Results"])

# Input tab
with input_tab:
    # Demo templates selector
    st.subheader("🧪 Quick Demo Templates")
    demo_template = st.selectbox(
        "Select a demo scenario",
        list(demo_templates.keys()),
        index=0
    )
    
    if st.button("Load Demo Template"):
        selected_template = demo_templates[demo_template]
        st.session_state.customer_data = selected_template["customer_data"]
        st.session_state.call_transcript = selected_template["call_transcript"]
        st.session_state.travel_notice_data = selected_template["travel_notice_data"]
        st.session_state.recent_transaction = selected_template["recent_transaction"]
        st.success(f"Loaded template: {demo_template}")
    
    st.markdown("---")
    
    # Custom input
    st.subheader("📋 Custom Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**👤 Customer Information**")
        customer_name = st.text_input("Customer Name", value=st.session_state.customer_data.get("name", ""))
        account_type = st.selectbox(
            "Account Type",
            ["Premium Checking", "Everyday Checking", "Savings Plus", "Student Account", "Business Account"],
            index=0 if st.session_state.customer_data.get("account_type") == "Premium Checking" else 1
        )
        credit_score = st.slider("Credit Score", 300, 850, st.session_state.customer_data.get("credit_score", 700))
        average_balance = st.text_input("Average Balance", value=st.session_state.customer_data.get("average_balance", "$5,000"))
        card_type = st.text_input("Card Type", value=st.session_state.customer_data.get("card_type", "Standard Visa"))
        eligible_for_upgrade = st.checkbox("Eligible for Upgrade", value=st.session_state.customer_data.get("eligible_for_upgrade", False))
        contact_preference = st.selectbox(
            "Contact Preference",
            ["Mobile", "Email", "Mail"],
            index=0 if st.session_state.customer_data.get("contact_preference") == "Mobile" else 1
        )
        
        # Update customer data in session state
        if st.button("Update Customer Info"):
            st.session_state.customer_data = {
                "name": customer_name,
                "account_type": account_type,
                "account_opened": st.session_state.customer_data.get("account_opened", "January 1, 2020"),
                "credit_score": credit_score,
                "average_balance": average_balance,
                "card_type": card_type,
                "eligible_for_upgrade": eligible_for_upgrade,
                "contact_preference": contact_preference,
                "phone": st.session_state.customer_data.get("phone", "+1 (555) 123-4567"),
                "email": st.session_state.customer_data.get("email", "customer@email.com")
            }
            st.success("Customer information updated")
    
    with col2:
        st.write("**📱 Transaction & Travel Notice**")
        transaction_date = st.date_input("Transaction Date", value=datetime.now())
        merchant = st.text_input("Merchant", value=st.session_state.recent_transaction.get("merchant", "Sample Merchant"))
        location = st.text_input("Location", value=st.session_state.recent_transaction.get("location", "New York, USA"))
        amount = st.text_input("Amount", value=st.session_state.recent_transaction.get("amount", "$100.00"))
        transaction_status = st.selectbox(
            "Transaction Status",
            ["Completed", "Declined", "Pending", "Disputed"],
            index=0 if st.session_state.recent_transaction.get("status") == "Completed" else 1
        )
        
        # Update transaction in session state
        if st.button("Update Transaction"):
            st.session_state.recent_transaction = {
                "date": transaction_date.strftime("%B %d, %Y"),
                "merchant": merchant,
                "location": location,
                "amount": amount,
                "status": transaction_status,
                "reason": st.session_state.recent_transaction.get("reason", "N/A"),
                "card_used": f"{card_type} ending in {random.randint(1000, 9999)}"
            }
            st.success("Transaction information updated")
    
    st.write("**📞 Call Transcript or Customer Message**")
    tone_options = st.multiselect(
        "Customer Tone/Emotion Hints (for AI analysis)",
        ["Frustrated", "Angry", "Confused", "Worried", "Satisfied", "Appreciative", "Urgent", "Calm"],
        default=[]
    )
    
    call_transcript = st.text_area(
        "Transcript/Message Content",
        value=st.session_state.call_transcript,
        height=250
    )
    
    # Update transcript in session state
    if st.button("Update Transcript"):
        if tone_options:
            tone_hint = f"[Tone detected: {', '.join(tone_options)}]\n\n"
            st.session_state.call_transcript = tone_hint + call_transcript
        else:
            st.session_state.call_transcript = call_transcript
        st.success("Transcript updated")
    
    st.markdown("---")
    
    # Run analysis button
    if st.button("🔍 Run AI Analysis", use_container_width=True):
        if not st.session_state.api_key_set:
            st.error("Please enter your OpenAI API key in the sidebar first")
        else:
            with st.spinner("AI is analyzing the customer interaction..."):
                # Simulate loading time with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                sentiment_result, recommended_actions, chain_of_thought = analyze_with_ai(
                    st.session_state.call_transcript,
                    st.session_state.customer_data,
                    st.session_state.travel_notice_data,
                    st.session_state.recent_transaction,
                    model=model_option
                )
                
                st.session_state.sentiment_result = sentiment_result
                st.session_state.recommended_actions = recommended_actions
                st.session_state.chain_of_thought = chain_of_thought
                st.session_state.analyzed = True
                
                st.success("Analysis complete! Switch to the Results tab to see recommendations.")

# Results tab
with results_tab:
    if not st.session_state.analyzed:
        st.info("Run the analysis in the Input Data tab to see results here.")
    else:
        # Display customer and transaction details in side panel
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("👤 Customer Summary")
            st.write(f"**Name:** {st.session_state.customer_data.get('name', 'N/A')}")
            st.write(f"**Account:** {st.session_state.customer_data.get('account_type', 'N/A')}")
            st.write(f"**Card:** {st.session_state.customer_data.get('card_type', 'N/A')}")
            
            # Create credit score gauge
            credit_score = st.session_state.customer_data.get('credit_score', 0)
            fig = px.bar(
                x=["Credit Score"], 
                y=[credit_score],
                color=[credit_score],
                color_continuous_scale=[(0, "red"), (0.4, "yellow"), (0.7, "green"), (1, "green")],
                range_color=[300, 850],
                height=150
            )
            fig.update_layout(
                title_text=f"Credit Score: {credit_score}",
                xaxis_title=None,
                yaxis_title=None,
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💳 Recent Transaction")
            st.write(f"**Date:** {st.session_state.recent_transaction.get('date', 'N/A')}")
            st.write(f"**Merchant:** {st.session_state.recent_transaction.get('merchant', 'N/A')}")
            st.write(f"**Amount:** {st.session_state.recent_transaction.get('amount', 'N/A')}")
            status = st.session_state.recent_transaction.get('status', 'N/A')
            status_color = "red" if status == "Declined" else "green" if status == "Completed" else "orange"
            st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold'>{status}</span>", unsafe_allow_html=True)
        
        with col2:
            # Sentiment Analysis
            if st.session_state.sentiment_result:
                st.subheader("😊 Sentiment Analysis")
                sentiment = st.session_state.sentiment_result.get("sentiment", "NEUTRAL")
                confidence = st.session_state.sentiment_result.get("confidence", 0.5)
                emotions = st.session_state.sentiment_result.get("emotions", [])
                frustration_points = st.session_state.sentiment_result.get("frustration_points", [])
                
                sentiment_class = "sentiment-positive" if sentiment == "POSITIVE" else "sentiment-negative" if sentiment == "NEGATIVE" else "sentiment-neutral"
                sentiment_emoji = "😊" if sentiment == "POSITIVE" else "😡" if sentiment == "NEGATIVE" else "😐"
                
                st.markdown(f"<h3 class='{sentiment_class}'>{sentiment_emoji} {sentiment}</h3>", unsafe_allow_html=True)
                st.progress(confidence)
                st.write(f"Confidence: {confidence:.2f}")
                
                # Emotions detected
                if emotions:
                    st.write("**Key Emotions Detected:**")
                    emotions_str = ", ".join(emotions)
                    st.write(emotions_str)
                
                # Frustration points
                if frustration_points and sentiment == "NEGATIVE":
                    st.write("**Key Frustration Points:**")
                    for point in frustration_points:
                        st.markdown(f"- {point}")
            
            # Call transcript snippet
            st.subheader("📞 Call Transcript Snippet")
            
            # Show only a snippet of the transcript in the results view
            transcript = st.session_state.call_transcript
            if len(transcript) > 200:
                transcript = transcript[:200] + "..."
            
            st.text_area("", transcript, height=150, disabled=True)
            
            if st.button("View Full Transcript"):
                st.text_area("Full Transcript", st.session_state.call_transcript, height=300, disabled=True)
        
        st.markdown("---")
        
        # Recommended Next Best Actions
        st.header("🎯 Recommended Next Best Actions")
        
        if st.session_state.recommended_actions:
            # Group actions by priority
            high_priority = [a for a in st.session_state.recommended_actions if a.get("priority") == "High"]
            medium_priority = [a for a in st.session_state.recommended_actions if a.get("priority") == "Medium"]
            low_priority = [a for a in st.session_state.recommended_actions if a.get("priority") == "Low"]
            
            # Display the actions in priority order
            for action in high_priority + medium_priority + low_priority:
                with st.container():
                    st.markdown(f"""
                    <div class="action-card">
                        <h3>{action.get('icon', '🔹')} {action.get('action', 'Action')}</h3>
                        <p>{action.get('description', '')}</p>
                        <p>Priority: <span class="priority-{action.get('priority', 'Medium').lower()}">{action.get('priority', 'Medium')}</span></p>
                        <p>Category: {action.get('category', 'General')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chain of Thought Analysis
        if st.session_state.chain_of_thought:
            st.markdown("---")
            with st.expander("🧠 AI Chain of Thought Analysis", expanded=False):
                st.markdown(st.session_state.chain_of_thought)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Analysis", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"nba_analysis_{timestamp}.json"
                
                analysis_data = {
                    "timestamp": datetime.now().isoformat(),
                    "customer_data": st.session_state.customer_data,
                    "recent_transaction": st.session_state.recent_transaction,
                    "travel_notice_data": st.session_state.travel_notice_data,
                    "sentiment_result": st.session_state.sentiment_result,
                    "recommended_actions": st.session_state.recommended_actions,
                    "chain_of_thought": st.session_state.chain_of_thought
                }
                
                st.download_button(
                    label="Download Analysis",
                    data=json.dumps(analysis_data, indent=2),
                    file_name=filename,
                    mime="application/json",
                )
        
        with col2:
            if st.button("📊 Generate Report", use_container_width=True):
                st.info("Report generation feature will be available in the next update")
        
        with col3:
            if st.button("🔄 New Analysis", use_container_width=True):
                # Reset analysis results but keep input data
                st.session_state.analyzed = False
                st.session_state.sentiment_result = {}
                st.session_state.recommended_actions = []
                st.session_state.chain_of_thought = ""
                st.rerun()

# Add footer
st.markdown("---")
st.caption("Next Best Action Recommendation Engine - Enterprise v2.0")