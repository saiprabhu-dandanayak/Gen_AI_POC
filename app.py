# import streamlit as st
# import openai
# import pandas as pd
# import plotly.express as px
# import random
# import json
# from datetime import datetime
# import time
# import logging
# from constants import STYLES , DEMO_TEMPLATES

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Set page configuration
# st.set_page_config(
#     page_title="Next Best Action Recommendation Engine",
#     page_icon="🎯",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Apply custom CSS for better UI
# st.markdown(STYLES, unsafe_allow_html=True)

# # Initialize session state
# if 'analyzed' not in st.session_state:
#     st.session_state.analyzed = False
# if 'customer_data' not in st.session_state:
#     st.session_state.customer_data = {}
# if 'call_transcript' not in st.session_state:
#     st.session_state.call_transcript = ""
# if 'travel_notice_data' not in st.session_state:
#     st.session_state.travel_notice_data = {}
# if 'recent_transaction' not in st.session_state:
#     st.session_state.recent_transaction = {}
# if 'recommended_actions' not in st.session_state:
#     st.session_state.recommended_actions = []
# if 'sentiment_result' not in st.session_state:
#     st.session_state.sentiment_result = {}
# if 'chain_of_thought' not in st.session_state:
#     st.session_state.chain_of_thought = ""
# if 'api_key_set' not in st.session_state:
#     st.session_state.api_key_set = False

# # Main title and description
# st.markdown("<h1 class='main-header'>🎯 Next Best Action Recommendation Engine</h1>", unsafe_allow_html=True)
# st.markdown("<p class='sub-header'>Customer Support Analysis & AI-Powered Recommendation System</p>", unsafe_allow_html=True)
# st.markdown("---")

# # OpenAI API key input in sidebar
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     openai_api_key = st.text_input("OpenAI API Key", type="password")
    
#     if openai_api_key:
#         openai.api_key = openai_api_key
#         st.session_state.api_key_set = True
    
#     model_option = st.selectbox(
#         "Select AI Model",
#         ["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-turbo"],  # Updated model names
#     )
    
#     st.markdown("---")

# # Function to analyze sentiment and generate recommendations using OpenAI
# def analyze_with_ai(transcript, customer_data, travel_notice, transaction, model="gpt-3.5-turbo-1106"):
#     try:
#         client = openai.OpenAI(api_key=openai.api_key)  # Updated for new API client
        
#         # Prompt for sentiment analysis
#         sentiment_prompt = f"""
#         Analyze this customer input and automatically detect:
#         1. Overall sentiment (POSITIVE, NEGATIVE, or NEUTRAL)
#         2. Confidence score (0-1)
#         3. Specific emotions present (e.g., frustration, anger, satisfaction)
#         4. Key points of concern or satisfaction
        
#         Input:
#         {transcript}
        
#         Return only JSON with keys: sentiment, confidence, emotions, key_points
#         """
        
#         sentiment_response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are an expert in emotional analysis and customer service."},
#                 {"role": "user", "content": sentiment_prompt}
#             ],
#             temperature=0,
#             max_tokens=500
#         )
        
#         sentiment_content = sentiment_response.choices[0].message.content.strip()
#         if not sentiment_content:
#             raise ValueError("Empty response from API")
#         sentiment_result = json.loads(sentiment_content)
        
#         # Prompt for next best action recommendations
#         action_prompt = f"""
#         Based on the following information, recommend the next best actions for a bank agent to take:

#         CUSTOMER INFORMATION:
#         {json.dumps(customer_data, indent=2)}

#         RECENT TRANSACTION:
#         {json.dumps(transaction, indent=2)}

#         TRAVEL NOTICE:
#         {json.dumps(travel_notice, indent=2)}

#         CALL TRANSCRIPT:
#         {transcript}

#         SENTIMENT ANALYSIS:
#         {json.dumps(sentiment_result, indent=2)}

#         Return a JSON array with the top 5 recommended actions, each containing:
#         1. "action": a short action title
#         2. "description": detailed description of what the agent should do
#         3. "priority": "High", "Medium", or "Low"
#         4. "icon": an appropriate emoji for the action
#         5. "category": the category of action (e.g., "Technical Resolution", "Customer Service", "Sales Opportunity")
        
#         Return only the JSON array with no additional text.
#         """
        
#         action_response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are an expert in banking customer service strategy."},
#                 {"role": "user", "content": action_prompt}
#             ],
#             temperature=0.2,
#             max_tokens=1000
#         )
        
#         action_content = action_response.choices[0].message.content.strip()
#         if not action_content:
#             raise ValueError("Empty response from API")
#         recommended_actions = json.loads(action_content)
        
#         # Prompt for reasoning chain of thought
#         reasoning_prompt = f"""
#         Provide a detailed step-by-step chain of thought analysis for this customer service situation:

#         CUSTOMER INFORMATION:
#         {json.dumps(customer_data, indent=2)}

#         RECENT TRANSACTION:
#         {json.dumps(transaction, indent=2)}

#         TRAVEL NOTICE:
#         {json.dumps(travel_notice, indent=2)}

#         CALL TRANSCRIPT:
#         {transcript}

#         SENTIMENT ANALYSIS:
#         {json.dumps(sentiment_result, indent=2)}

#         Include sections for:
#         1. Customer Context Analysis
#         2. Problem Identification
#         3. Emotional Impact Assessment
#         4. Priority Determination
#         5. Opportunity Analysis
#         6. Long-term Relationship Considerations

#         Give a thorough, thoughtful analysis that demonstrates expert-level understanding of banking customer service.
#         """
        
#         reasoning_response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are an expert in customer psychology and banking operations."},
#                 {"role": "user", "content": reasoning_prompt}
#             ],
#             temperature=0.3,
#             max_tokens=1500
#         )
#         print("🚀 ~ reasoning_response:", reasoning_response) 
#         chain_of_thought = reasoning_response.choices[0].message.content
#         print("🚀 ~ chain_of_thought:", chain_of_thought)
        
#         return sentiment_result, recommended_actions, chain_of_thought
    
#     except json.JSONDecodeError as e:
#         logger.error(f"JSON parsing error: {str(e)}")
#         st.error(f"Error parsing API response: {str(e)}")
#         return {
#             "sentiment": "ERROR",
#             "confidence": 0,
#             "emotions": ["Error in analysis"],
#             "key_points": ["Invalid response format"]
#         }, [], "Error generating analysis"
#     except Exception as e:
#         logger.error(f"API call failed: {str(e)}")
#         st.error(f"Error calling OpenAI API: {str(e)}")
#         return {
#             "sentiment": "ERROR",
#             "confidence": 0,
#             "emotions": ["Error in analysis"],
#             "key_points": ["API error occurred"]
#         }, [], "Error generating analysis"

# # Tabs for Input and Results
# input_tab, results_tab = st.tabs(["📝 Input Data", "📊 Analysis Results"])

# # Input tab
# with input_tab:
#     # Demo templates selector
#     st.subheader("🧪 Quick Demo Templates")
#     demo_template = st.selectbox(
#         "Select a demo scenario",
#         list(DEMO_TEMPLATES.keys()),
#         index=0
#     )
    
#     if st.button("Load Demo Template"):
#         selected_template = DEMO_TEMPLATES[demo_template]
#         st.session_state.customer_data = selected_template["customer_data"]
#         st.session_state.call_transcript = selected_template["call_transcript"]
#         st.session_state.travel_notice_data = selected_template["travel_notice_data"]
#         st.session_state.recent_transaction = selected_template["recent_transaction"]
#         st.success(f"Loaded template: {demo_template}")
    
#     st.markdown("---")
    
#     # Custom input
#     st.subheader("📋 Custom Input")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.write("**👤 Customer Information**")
#         customer_name = st.text_input("Customer Name", value=st.session_state.customer_data.get("name", ""))
#         account_type = st.selectbox(
#             "Account Type",
#             ["Premium Checking", "Everyday Checking", "Savings Plus", "Student Account", "Business Account"],
#             index=0 if st.session_state.customer_data.get("account_type") == "Premium Checking" else 1
#         )
#         credit_score = st.slider("Credit Score", 300, 850, st.session_state.customer_data.get("credit_score", 700))
#         average_balance = st.text_input("Average Balance", value=st.session_state.customer_data.get("average_balance", "$5,000"))
#         card_type = st.text_input("Card Type", value=st.session_state.customer_data.get("card_type", "Standard Visa"))
#         eligible_for_upgrade = st.checkbox("Eligible for Upgrade", value=st.session_state.customer_data.get("eligible_for_upgrade", False))
#         contact_preference = st.selectbox(
#             "Contact Preference",
#             ["Mobile", "Email", "Mail"],
#             index=0 if st.session_state.customer_data.get("contact_preference") == "Mobile" else 1
#         )
        
#         # Update customer data in session state
#         if st.button("Update Customer Info"):
#             st.session_state.customer_data = {
#                 "name": customer_name,
#                 "account_type": account_type,
#                 "account_opened": st.session_state.customer_data.get("account_opened", "January 1, 2020"),
#                 "credit_score": credit_score,
#                 "average_balance": average_balance,
#                 "card_type": card_type,
#                 "eligible_for_upgrade": eligible_for_upgrade,
#                 "contact_preference": contact_preference,
#                 "phone": st.session_state.customer_data.get("phone", "+1 (555) 123-4567"),
#                 "email": st.session_state.customer_data.get("email", "customer@email.com")
#             }
#             st.success("Customer information updated")
    
#     with col2:
#         st.write("**📱 Transaction & Travel Notice**")
#         transaction_date = st.date_input("Transaction Date", value=datetime.now())
#         merchant = st.text_input("Merchant", value=st.session_state.recent_transaction.get("merchant", "Sample Merchant"))
#         location = st.text_input("Location", value=st.session_state.recent_transaction.get("location", "New York, USA"))
#         amount = st.text_input("Amount", value=st.session_state.recent_transaction.get("amount", "$100.00"))
#         transaction_status = st.selectbox(
#             "Transaction Status",
#             ["Completed", "Declined", "Pending", "Disputed"],
#             index=0 if st.session_state.recent_transaction.get("status") == "Completed" else 1
#         )
        
#         # Update transaction in session state
#         if st.button("Update Transaction"):
#             st.session_state.recent_transaction = {
#                 "date": transaction_date.strftime("%B %d, %Y"),
#                 "merchant": merchant,
#                 "location": location,
#                 "amount": amount,
#                 "status": transaction_status,
#                 "reason": st.session_state.recent_transaction.get("reason", "N/A"),
#                 "card_used": f"{card_type} ending in {random.randint(1000, 9999)}"
#             }
#             st.success("Transaction information updated")
    
#     # Input method selection
#     st.write("**📞 Customer Input Method**")
#     input_method = st.radio(
#         "Choose input method",
#         ["Text Message", "Voice Note"],
#         horizontal=True
#     )
    
#     if input_method == "Text Message":
#         st.write("**💬 Customer Message**")
#         call_transcript = st.text_area(
#             "Type the customer's message",
#             value=st.session_state.call_transcript,
#             height=250,
#             placeholder="Enter customer's message here..."
#         )
        
#         if st.button("Update Message"):
#             st.session_state.call_transcript = call_transcript
#             st.success("Message updated")
            
#     else:  # Voice Note
#         st.write("**🎙️ Voice Note Input**")
#         st.info("Voice input simulation: Upload an audio file or type text to simulate voice transcription")
        
#         # Audio upload option
#         audio_file = st.file_uploader("Upload voice note (MP3/WAV)", type=["mp3", "wav"])
        
#         # Alternative text input to simulate transcription
#         voice_transcript = st.text_area(
#             "Or type to simulate voice transcription",
#             value=st.session_state.call_transcript,
#             height=150,
#             placeholder="Type what would be said in the voice note..."
#         )
        
#         if st.button("Process Voice Input"):
#             if audio_file:
#                 # Simulate audio processing
#                 st.session_state.call_transcript = "Audio processing not implemented. Please use text simulation."
#                 st.warning("Audio processing is a simulation only in this version")
#             elif voice_transcript:
#                 st.session_state.call_transcript = voice_transcript
#                 st.success("Voice simulation updated")
#             else:
#                 st.error("Please upload an audio file or type a simulation")

#     st.markdown("---")
    
#     # Run analysis button
#     if st.button("🔍 Run AI Analysis", use_container_width=True):
#         if not st.session_state.api_key_set:
#             st.error("Please enter your OpenAI API key in the sidebar first")
#         else:
#             with st.spinner("AI is analyzing the customer input and detecting emotions..."):
#                 progress_bar = st.progress(0)
#                 for i in range(100):
#                     time.sleep(0.02)
#                     progress_bar.progress(i + 1)
                
#                 sentiment_result, recommended_actions, chain_of_thought = analyze_with_ai(
#                     st.session_state.call_transcript,
#                     st.session_state.customer_data,
#                     st.session_state.travel_notice_data,
#                     st.session_state.recent_transaction,
#                     model=model_option
#                 )
                
#                 st.session_state.sentiment_result = sentiment_result
#                 st.session_state.recommended_actions = recommended_actions
#                 st.session_state.chain_of_thought = chain_of_thought
#                 st.session_state.analyzed = True
                
#                 st.success("Analysis complete! Switch to the Results tab to see recommendations and detected emotions.")

# # Results tab
# with results_tab:
#     if not st.session_state.analyzed:
#         st.info("Run the analysis in the Input Data tab to see results here.")
#     else:
#         # Display customer and transaction details in side panel
#         col1, col2 = st.columns([2, 3])
        
#         with col1:
#             st.subheader("👤 Customer Summary")
#             st.write(f"**Name:** {st.session_state.customer_data.get('name', 'N/A')}")
#             st.write(f"**Account:** {st.session_state.customer_data.get('account_type', 'N/A')}")
#             st.write(f"**Card:** {st.session_state.customer_data.get('card_type', 'N/A')}")
            
#             # Create credit score gauge
#             credit_score = st.session_state.customer_data.get('credit_score', 0)
#             fig = px.bar(
#                 x=["Credit Score"], 
#                 y=[credit_score],
#                 color=[credit_score],
#                 color_continuous_scale=[(0, "red"), (0.4, "yellow"), (0.7, "green"), (1, "green")],
#                 range_color=[300, 850],
#                 height=150
#             )
#             fig.update_layout(
#                 title_text=f"Credit Score: {credit_score}",
#                 xaxis_title=None,
#                 yaxis_title=None,
#                 coloraxis_showscale=False,
#                 margin=dict(l=20, r=20, t=40, b=20)
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#             st.subheader("💳 Recent Transaction")
#             st.write(f"**Date:** {st.session_state.recent_transaction.get('date', 'N/A')}")
#             st.write(f"**Merchant:** {st.session_state.recent_transaction.get('merchant', 'N/A')}")
#             st.write(f"**Amount:** {st.session_state.recent_transaction.get('amount', 'N/A')}")
#             status = st.session_state.recent_transaction.get('status', 'N/A')
#             status_color = "red" if status == "Declined" else "green" if status == "Completed" else "orange"
#             st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold'>{status}</span>", unsafe_allow_html=True)
        
#         with col2:
#             # Sentiment Analysis
#             if st.session_state.sentiment_result:
#                 st.subheader("😊 Sentiment Analysis")
#                 sentiment = st.session_state.sentiment_result.get("sentiment", "NEUTRAL")
#                 confidence = st.session_state.sentiment_result.get("confidence", 0.5)
#                 emotions = st.session_state.sentiment_result.get("emotions", [])
#                 key_points = st.session_state.sentiment_result.get("key_points", [])
                
#                 sentiment_class = "sentiment-positive" if sentiment == "POSITIVE" else "sentiment-negative" if sentiment == "NEGATIVE" else "sentiment-neutral"
#                 sentiment_emoji = "😊" if sentiment == "POSITIVE" else "😡" if sentiment == "NEGATIVE" else "😐"
                
#                 st.markdown(f"<h3 class='{sentiment_class}'>{sentiment_emoji} {sentiment}</h3>", unsafe_allow_html=True)
#                 st.progress(confidence)
#                 st.write(f"Confidence: {confidence:.2f}")
                
#                 # Emotions detected
#                 if emotions:
#                     st.write("**Key Emotions Detected:**")
#                     emotions_str = ", ".join(emotions)
#                     st.write(emotions_str)
                
#                 # Key points
#                 if key_points:
#                     st.write("**Key Points of Concern/Satisfaction:**")
#                     for point in key_points:
#                         st.markdown(f"- {point}")
            
#             # Input content snippet
#             st.subheader("📞 Input Content Snippet")
            
#             # Show only a snippet of the input
#             transcript = st.session_state.call_transcript
#             if len(transcript) > 200:
#                 transcript = transcript[:200] + "..."
            
#             st.text_area("Content Preview", transcript, height=150, disabled=True)
            
#             if st.button("View Full Content"):
#                 st.text_area("Full Content", st.session_state.call_transcript, height=300, disabled=True)
        
#         st.markdown("---")
        
#         # Recommended Next Best Actions
#         st.header("🎯 Recommended Next Best Actions")
        
#         if st.session_state.recommended_actions:
#             # Group actions by priority
#             high_priority = [a for a in st.session_state.recommended_actions if a.get("priority") == "High"]
#             medium_priority = [a for a in st.session_state.recommended_actions if a.get("priority") == "Medium"]
#             low_priority = [a for a in st.session_state.recommended_actions if a.get("priority") == "Low"]
            
#             # Display the actions in priority order
#             for action in high_priority + medium_priority + low_priority:
#                 with st.container():
#                     st.markdown(f"""
#                     <div class="action-card">
#                         <h3>{action.get('icon', '🔹')} {action.get('action', 'Action')}</h3>
#                         <p>{action.get('description', '')}</p>
#                         <p>Priority: <span class="priority-{action.get('priority', 'Medium').lower()}">{action.get('priority', 'Medium')}</span></p>
#                         <p>Category: {action.get('category', 'General')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
        
#         # Chain of Thought Analysis
#         if st.session_state.chain_of_thought:
#             st.markdown("---")
#             with st.expander("🧠 AI Chain of Thought Analysis", expanded=False):
#                 st.markdown(st.session_state.chain_of_thought)
        
#         # Action buttons
#         st.markdown("---")
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             if st.button("💾 Save Analysis", use_container_width=True):
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"nba_analysis_{timestamp}.json"
                
#                 analysis_data = {
#                     "timestamp": datetime.now().isoformat(),
#                     "customer_data": st.session_state.customer_data,
#                     "recent_transaction": st.session_state.recent_transaction,
#                     "travel_notice_data": st.session_state.travel_notice_data,
#                     "sentiment_result": st.session_state.sentiment_result,
#                     "recommended_actions": st.session_state.recommended_actions,
#                     "chain_of_thought": st.session_state.chain_of_thought
#                 }
                
#                 st.download_button(
#                     label="Download Analysis",
#                     data=json.dumps(analysis_data, indent=2),
#                     file_name=filename,
#                     mime="application/json",
#                 )
        
#         with col2:
#             if st.button("📊 Generate Report", use_container_width=True):
#                 st.info("Report generation feature will be available in the next update")
        
#         with col3:
#             if st.button("🔄 New Analysis", use_container_width=True):
#                 # Reset analysis results but keep input data
#                 st.session_state.analyzed = False
#                 st.session_state.sentiment_result = {}
#                 st.session_state.recommended_actions = []
#                 st.session_state.chain_of_thought = ""
#                 st.rerun()

# # Add footer
# st.markdown("---")
# st.caption("Next Best Action Recommendation Engine - Enterprise v2.0")

import streamlit as st
import openai
import pandas as pd
import plotly.express as px
import random
import json
from datetime import datetime
import time
import logging
from constants import STYLES, DEMO_TEMPLATES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Next Best Action Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown(STYLES, unsafe_allow_html=True)

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
if 'template_loaded' not in st.session_state:
    st.session_state.template_loaded = False

# Auto-load the first template if no template is loaded
if not st.session_state.template_loaded:
    first_template_key = list(DEMO_TEMPLATES.keys())[0]  # Get the first template key
    selected_template = DEMO_TEMPLATES[first_template_key]
    
    # Load the template data into session state
    st.session_state.customer_data = selected_template["customer_data"]
    st.session_state.call_transcript = selected_template["call_transcript"]
    st.session_state.travel_notice_data = selected_template["travel_notice_data"]
    st.session_state.recent_transaction = selected_template["recent_transaction"]
    st.session_state.template_loaded = True

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
        ["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-turbo"],  # Updated model names
    )
    
    st.markdown("---")

# Function to analyze sentiment and generate recommendations using OpenAI
def analyze_with_ai(transcript, customer_data, travel_notice, transaction, model="gpt-3.5-turbo-1106"):
    try:
        client = openai.OpenAI(api_key=openai.api_key)  # Updated for new API client
        
        # Prompt for sentiment analysis
        sentiment_prompt = f"""
        Analyze this customer input and automatically detect:
        1. Overall sentiment (POSITIVE, NEGATIVE, or NEUTRAL)
        2. Confidence score (0-1)
        3. Specific emotions present (e.g., frustration, anger, satisfaction)
        4. Key points of concern or satisfaction
        
        Input:
        {transcript}
        
        Return only JSON with keys: sentiment, confidence, emotions, key_points
        """
        
        sentiment_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in emotional analysis and customer service."},
                {"role": "user", "content": sentiment_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        sentiment_content = sentiment_response.choices[0].message.content.strip()
        if not sentiment_content:
            raise ValueError("Empty response from API")
        sentiment_result = json.loads(sentiment_content)
        
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
        
        action_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in banking customer service strategy."},
                {"role": "user", "content": action_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        action_content = action_response.choices[0].message.content.strip()
        if not action_content:
            raise ValueError("Empty response from API")
        recommended_actions = json.loads(action_content)
        
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
        
        reasoning_response = client.chat.completions.create(
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
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        st.error(f"Error parsing API response: {str(e)}")
        return {
            "sentiment": "ERROR",
            "confidence": 0,
            "emotions": ["Error in analysis"],
            "key_points": ["Invalid response format"]
        }, [], "Error generating analysis"
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        st.error(f"Error calling OpenAI API: {str(e)}")
        return {
            "sentiment": "ERROR",
            "confidence": 0,
            "emotions": ["Error in analysis"],
            "key_points": ["API error occurred"]
        }, [], "Error generating analysis"

# Tabs for Input and Results
input_tab, results_tab = st.tabs(["📝 Input Data", "📊 Analysis Results"])

# Input tab
with input_tab:
    # Demo templates selector
    st.subheader("🧪 Quick Demo Templates")
    demo_template = st.selectbox(
        "Select a demo scenario",
        list(DEMO_TEMPLATES.keys()),
        index=0
    )
    
    if st.button("Load Demo Template"):
        selected_template = DEMO_TEMPLATES[demo_template]
        st.session_state.customer_data = selected_template["customer_data"]
        st.session_state.call_transcript = selected_template["call_transcript"]
        st.session_state.travel_notice_data = selected_template["travel_notice_data"]
        st.session_state.recent_transaction = selected_template["recent_transaction"]
        st.success(f"Loaded template: {demo_template}")
    
    st.markdown("---")
    
    # Custom input (read-only customer details)
    st.subheader("📋 Customer Information (Read-Only)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**👤 Customer Details**")
        
        # Display customer information in read-only format
        if st.session_state.customer_data:
            for key, value in st.session_state.customer_data.items():
                if key not in ['credit_score', 'eligible_for_upgrade']:  # Handle these separately
                    # Convert snake_case to Title Case for display
                    display_key = ' '.join(word.capitalize() for word in key.split('_'))
                    st.write(f"**{display_key}:** {value}")
                    
            # Display credit score as a metric
            st.metric("Credit Score", st.session_state.customer_data.get('credit_score', 'N/A'))
            
            # Display eligible for upgrade as a checkbox (disabled)
            st.write("**Eligible for Upgrade:** ", 
                   "✅ Yes" if st.session_state.customer_data.get('eligible_for_upgrade', False) else "❌ No")
        else:
            st.info("Please load a template to view customer information")
    
    with col2:
        st.write("**📱 Travel Notice Information**")
        
        # Display travel notice information in read-only format
        if st.session_state.travel_notice_data:
            for key, value in st.session_state.travel_notice_data.items():
                # Handle special case for countries list
                if key == 'countries' and isinstance(value, list):
                    st.write(f"**Countries:** {', '.join(value)}")
                else:
                    # Convert snake_case to Title Case for display
                    display_key = ' '.join(word.capitalize() for word in key.split('_'))
                    st.write(f"**{display_key}:** {value}")
        else:
            st.info("Please load a template to view travel notice information")
    
    # Recent Transactions display
    st.subheader("💳 Recent Transactions")
    if isinstance(st.session_state.recent_transaction, list) and len(st.session_state.recent_transaction) > 0:
        # Create a transaction history display with improved UI
        for i, transaction in enumerate(st.session_state.recent_transaction[:3]):  # Display only the last 3 transactions
            # Determine status color
            status = transaction.get('status', 'Unknown')
            status_color = "red" if status == "Declined" else "green" if status == "Approved" else "orange"
            
            # Create a nice transaction card with CSS
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 10px;  margin-bottom: 10px; border-left: 5px solid  
                {'#D32F2F' if status == 'Declined' else '#4CAF50' if status == 'Approved' else '#FF9800'}" height:25px>
                <div style="display: flex; justify-content: space-between;">
                    <h4 style="margin: 0; font-size: 0.95rem;">{transaction.get('merchant', 'Unknown Merchant')}</h4>
                    <div>
                        <span style="color: {status_color}; font-weight: bold;">{status}</span>
                    </div>
                </div>
                <div style="margin: 3px 0; line-height: 1.2;">
                    <strong>Date:</strong> {transaction.get('date', 'Unknown')} | 
                    <strong>Amount:</strong> {transaction.get('amount', 'Unknown')} | 
                    <strong>Location:</strong> {transaction.get('location', 'Unknown')}
                </div>
                <div style="margin: 3px 0; line-height: 1.2;">
                    <strong>Card:</strong> {transaction.get('card_used', 'Unknown')} 
                    {f"| <strong>Reason:</strong> {transaction.get('reason')}" if 'reason' in transaction and transaction['reason'] != "N/A" else ""}
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Please load a template to view recent transactions")
    
    # Input method selection
    st.write("**📞 Customer Input Method**")
    input_method = st.radio(
        "Choose input method",
        ["Text Message", "Voice Note"],
        horizontal=True
    )
    
    if input_method == "Text Message":
        st.write("**💬 Customer Message**")
        call_transcript = st.text_area(
            "Type the customer's message",
            value=st.session_state.call_transcript,
            height=250,
            placeholder="Enter customer's message here..."
        )
        
        if st.button("Update Message"):
            st.session_state.call_transcript = call_transcript
            st.success("Message updated")
            
    else:  # Voice Note
        st.write("**🎙️ Voice Note Input**")
        st.info("Voice input simulation: Upload an audio file or type text to simulate voice transcription")
        
        # Audio upload option
        audio_file = st.file_uploader("Upload voice note (MP3/WAV)", type=["mp3", "wav"])
        
        # Alternative text input to simulate transcription
        voice_transcript = st.text_area(
            "Or type to simulate voice transcription",
            value=st.session_state.call_transcript,
            height=150,
            placeholder="Type what would be said in the voice note..."
        )
        
        if st.button("Process Voice Input"):
            if audio_file:
                # Simulate audio processing
                st.session_state.call_transcript = "Audio processing not implemented. Please use text simulation."
                st.warning("Audio processing is a simulation only in this version")
            elif voice_transcript:
                st.session_state.call_transcript = voice_transcript
                st.success("Voice simulation updated")
            else:
                st.error("Please upload an audio file or type a simulation")

    st.markdown("---")
    
    # Run analysis button
    if st.button("🔍 Run AI Analysis", use_container_width=True):
        if not st.session_state.api_key_set:
            st.error("Please enter your OpenAI API key in the sidebar first")
        else:
            with st.spinner("AI is analyzing the customer input and detecting emotions..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                # For analysis, get only the most recent transaction if we have a list
                recent_transaction_for_analysis = st.session_state.recent_transaction[0] if isinstance(st.session_state.recent_transaction, list) else st.session_state.recent_transaction
                
                sentiment_result, recommended_actions, chain_of_thought = analyze_with_ai(
                    st.session_state.call_transcript,
                    st.session_state.customer_data,
                    st.session_state.travel_notice_data,
                    recent_transaction_for_analysis,
                    model=model_option
                )
                
                st.session_state.sentiment_result = sentiment_result
                st.session_state.recommended_actions = recommended_actions
                st.session_state.chain_of_thought = chain_of_thought
                st.session_state.analyzed = True
                
                st.success("Analysis complete! Switch to the Results tab to see recommendations and detected emotions.")

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
            
            # Recent Transaction summary
            st.subheader("💳 Recent Transactions")
            
            # Display the 3 most recent transactions with improved UI
            if isinstance(st.session_state.recent_transaction, list) and len(st.session_state.recent_transaction) > 0:
                for i, transaction in enumerate(st.session_state.recent_transaction[:3]):
                    status = transaction.get('status', 'Unknown')
                    status_color = "red" if status == "Declined" else "green" if status == "Approved" else "orange"
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 10px; 
                              margin-bottom: 10px; border-left: 5px solid 
                              {'#D32F2F' if status == 'Declined' else '#4CAF50' if status == 'Approved' else '#FF9800'}">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{transaction.get('merchant', 'Unknown')}</strong>
                            <span style="color: {status_color}; font-weight: bold;">{status}</span>
                        </div>
                        <p style="margin: 2px 0; font-size: 0.9em;">
                            {transaction.get('date', 'Unknown')} | {transaction.get('amount', 'Unknown')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                recent_tx = st.session_state.recent_transaction
                status = recent_tx.get('status', 'N/A')
                status_color = "red" if status == "Declined" else "green" if status == "Completed" else "orange"
                st.write(f"**Date:** {recent_tx.get('date', 'N/A')}")
                st.write(f"**Merchant:** {recent_tx.get('merchant', 'N/A')}")
                st.write(f"**Amount:** {recent_tx.get('amount', 'N/A')}")
                st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold'>{status}</span>", unsafe_allow_html=True)
        
        with col2:
            # Sentiment Analysis
            if st.session_state.sentiment_result:
                st.subheader("😊 Sentiment Analysis")
                sentiment = st.session_state.sentiment_result.get("sentiment", "NEUTRAL")
                confidence = st.session_state.sentiment_result.get("confidence", 0.5)
                emotions = st.session_state.sentiment_result.get("emotions", [])
                key_points = st.session_state.sentiment_result.get("key_points", [])
                
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
                
                # Key points
                if key_points:
                    st.write("**Key Points of Concern/Satisfaction:**")
                    for point in key_points:
                        st.markdown(f"- {point}")
            
            # Input content snippet
            st.subheader("📞 Input Content Snippet")
            
            # Show only a snippet of the input
            transcript = st.session_state.call_transcript
            if len(transcript) > 200:
                transcript = transcript[:200] + "..."
            
            st.text_area("Content Preview", transcript, height=150, disabled=True)
            
            if st.button("View Full Content"):
                st.text_area("Full Content", st.session_state.call_transcript, height=300, disabled=True)
        
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