import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
import time
import logging
from constants import DEMO_TEMPLATES, sentiment_prompt, action_prompt, reasoning_prompt, STYLES, CHAT_BOX_STYLES
from agent_router import RouterAgent
from specialized_agents import get_agent_for_routing
import requests

# Configure logging with a custom handler for chain_of_thought
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ChainOfThought')

if "chain_of_thought" not in st.session_state:
    st.session_state.chain_of_thought = []



# Custom handler to append logs to chain_of_thought
class ChainOfThoughtHandler(logging.Handler):
    def __init__(self, update_callback=None):
        super().__init__()
        self.update_callback = update_callback

    def emit(self, record):
        log_message = self.format(record)
        # Assuming chain_of_thought is accessible via st.session_state
        if 'chain_of_thought' in st.session_state:
            st.session_state.chain_of_thought += f"{log_message}\n"
            if self.update_callback:
                self.update_callback(st.session_state.chain_of_thought)


# Add custom handler to logger
chain_handler = ChainOfThoughtHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chain_handler.setFormatter(formatter)
logger.addHandler(chain_handler)



def analyze_with_groq(transcript, customer_data, travel_notice_data, recent_transaction, model, groq_api_key, update_callback=None):
    sentiment_result = {}
    recommended_actions = []
    st.session_state.chain_of_thought = "Starting analysis...\n"
    chain_of_thought = st.session_state.chain_of_thought

    # Update logger's custom handler with the current callback
    for handler in logger.handlers:
        if isinstance(handler, ChainOfThoughtHandler):
            handler.update_callback = update_callback

    # Validate transcript
    if not transcript.strip():
        chain_of_thought += "Error: No customer input provided. Please enter a query in the Customer Chat to proceed with analysis.\n"
        logger.error("No customer input provided")
        if update_callback:
            update_callback(chain_of_thought)
        return sentiment_result, recommended_actions, chain_of_thought

    # Initialize RouterAgent with Groq API key
    chain_of_thought += "\n=== Routing Agent Analysis ===\n"
    chain_of_thought += "Initializing AI-driven RouterAgent to determine the appropriate specialized agent...\n"
    logger.info("Initializing RouterAgent")
    router = RouterAgent(customer_data, travel_notice_data, recent_transaction if isinstance(recent_transaction, list) else [recent_transaction], groq_api_key, model)
    
    # Detailed routing process
    chain_of_thought += f"User query received: '{transcript}'\n"
    chain_of_thought += "Step 1: Sending query to Groq API for AI-based routing...\n"
    logger.info(f"Routing query: {transcript}")
    selected_agent_name, routing_log = router.route(transcript)

    st.session_state.selected_agent = selected_agent_name
    
    # Log AI routing details
    chain_of_thought += "AI routing results:\n"
    chain_of_thought += f"- Selected agent: {selected_agent_name}\n"
    chain_of_thought += f"- AI reasoning: {routing_log.get('ai_reasoning', 'No AI reasoning provided')}\n"
    logger.info(f"Selected agent: {selected_agent_name}")
    chain_of_thought += "- Confidence scores:\n"
    confidence_scores = routing_log.get('confidence_scores', {})
    if confidence_scores:
        chain_of_thought += "  Calculation breakdown:\n"
        chain_of_thought += "  - Confidence scores are computed as a weighted combination of keyword matches, pattern matches, and context analysis.\n"
        chain_of_thought += "  - Weights (assumed typical values, may vary):\n"
        chain_of_thought += "    - Keyword matches: 40% (based on number and relevance of matched keywords)\n"
        chain_of_thought += "    - Pattern matches: 30% (based on number and specificity of regex patterns)\n"
        chain_of_thought += "    - Context analysis: 30% (based on relevance to recent transactions, travel notices, etc.)\n"
        for agent, score in confidence_scores.items():
            chain_of_thought += f"    - {agent}: {score:.2f}\n"
            keyword_count = len(routing_log.get('keyword_matches', {}).get(agent, []))
            pattern_count = len(routing_log.get('pattern_matches', {}).get(agent, []))
            context_score = routing_log.get('context_analysis', {}).get(agent, 0)
            chain_of_thought += f"      - Keywords matched: {keyword_count} (contributes to score)\n"
            chain_of_thought += f"      - Patterns matched: {pattern_count} (contributes to score)\n"
            chain_of_thought += f"      - Context relevance: {context_score:.2f} (based on transaction/travel data)\n"
            logger.debug(f"Agent {agent}: Keywords={keyword_count}, Patterns={pattern_count}, Context={context_score:.2f}")
    else:
        chain_of_thought += "  - No confidence scores provided.\n"
        logger.warning("No confidence scores provided")
    
    # Log rule-based analysis for transparency
    chain_of_thought += "\nSupplementary rule-based analysis (for transparency):\n"
    chain_of_thought += "Keyword matches found:\n"
    for agent, matches in routing_log.get('keyword_matches', {}).items():
        chain_of_thought += f"- {agent}: {', '.join(matches)}\n"
        logger.debug(f"Keyword matches for {agent}: {matches}")
    if not routing_log.get('keyword_matches'):
        chain_of_thought += "- None\n"
    
    chain_of_thought += "\nPattern matches found:\n"
    for agent, patterns in routing_log.get('pattern_matches', {}).items():
        chain_of_thought += f"- {agent}: {len(patterns)} pattern(s) matched\n"
        for pattern in patterns:
            chain_of_thought += f"  - Pattern: {pattern}\n"
            logger.debug(f"Pattern match for {agent}: {pattern}")
    if not routing_log.get('pattern_matches'):
        chain_of_thought += "- None\n"
    
    chain_of_thought += "\nContext analysis based on recent activity:\n"
    for agent, score in routing_log.get('context_analysis', {}).items():
        chain_of_thought += f"- {agent}: Score {score}\n"
        if agent == "TransactionAnalysisAgent" and score > 0:
            chain_of_thought += "  - Likely due to mentions of recent transaction merchants or locations\n"
        elif agent == "TravelNoticeAgent" and score > 0:
            chain_of_thought += "  - Likely due to mentions of travel-related locations or keywords\n"
        elif agent == "CardServicesAgent" and score > 0:
            chain_of_thought += "  - Likely due to mentions of card-specific issues like 'lost'\n"
        logger.debug(f"Context score for {agent}: {score}")
    if not routing_log.get('context_analysis'):
        chain_of_thought += "- No significant context clues found\n"
    
    chain_of_thought += f"\nRouting decision: {routing_log.get('routing_decision', 'No decision provided')}\n"
    logger.info(f"Routing decision: {routing_log.get('routing_decision', 'No decision provided')}")
    
    if update_callback:
        update_callback(chain_of_thought)
        st.session_state.chain_of_thought = chain_of_thought

    # Perform sentiment analysis
    chain_of_thought += "\n=== Sentiment Analysis ===\n"
    chain_of_thought += "Analyzing sentiment using Groq API...\n"
    logger.info("Starting sentiment analysis")
    sentiment_messages = [
        {"role": "system", "content": sentiment_prompt.format(transcript=transcript)}
    ]
    sentiment_response, sentiment_error = make_groq_request(sentiment_messages, model, groq_api_key)
    if sentiment_error:
        chain_of_thought += f"Sentiment analysis error: {sentiment_error}\n"
        logger.error(f"Sentiment analysis error: {sentiment_error}")
        sentiment_result = {"sentiment": "NEUTRAL", "confidence": 0.5, "emotions": [], "key_points": []}
    else:
        try:
            sentiment_result = json.loads(sentiment_response)
            chain_of_thought += f"Sentiment result: {json.dumps(sentiment_result, indent=2)}\n"
            logger.info(f"Sentiment result: {sentiment_result}")
        except json.JSONDecodeError:
            chain_of_thought += "Error: Invalid JSON response from sentiment analysis.\n"
            logger.error("Invalid JSON response from sentiment analysis")
            sentiment_result = {"sentiment": "NEUTRAL", "confidence": 0.5, "emotions": [], "key_points": []}
    
    if update_callback:
        update_callback(chain_of_thought)
        st.session_state.chain_of_thought = chain_of_thought

    # Get the specialized agent
    chain_of_thought += f"\n=== {selected_agent_name} Processing ===\n"
    chain_of_thought += f"Initializing {selected_agent_name} to process the query...\n"
    logger.info(f"Initializing {selected_agent_name}")
    agent = get_agent_for_routing(selected_agent_name, customer_data, travel_notice_data, recent_transaction if isinstance(recent_transaction, list) else [recent_transaction])
    
    # Process with the selected agent
    try:
        chain_of_thought += f"Processing query: '{transcript}'\n"
        logger.info(f"{selected_agent_name} processing query: {transcript}")
        agent_result = agent.process(transcript)
        
        # Log detailed agent reasoning
        reasoning_log = agent_result.get('reasoning_log', {})
        chain_of_thought += "\nDetailed agent reasoning:\n"
        
        chain_of_thought += "Analysis steps performed:\n"
        for step in reasoning_log.get('analysis_steps', []):
            chain_of_thought += f"- {step}\n"
            logger.debug(f"Agent step: {step}")
        
        chain_of_thought += "\nDecision factors considered:\n"
        for factor, value in reasoning_log.get('decision_factors', {}).items():
            chain_of_thought += f"- {factor}: {value}\n"
            logger.debug(f"Decision factor: {factor} = {value}")
        
        chain_of_thought += "\nActions considered:\n"
        for action in reasoning_log.get('actions_considered', []):
            chain_of_thought += f"- {action['action']} (Reason: {action['reason']})\n"
            logger.debug(f"Action considered: {action['action']}, Reason: {action['reason']}")
        
        chain_of_thought += "\nActions taken:\n"
        for action in reasoning_log.get('actions_taken', []):
            chain_of_thought += f"- {action['action']}: {action['details']}\n"
            logger.debug(f"Action taken: {action['action']}, Details: {action['details']}")
        
        chain_of_thought += f"\nResponse construction logic: {reasoning_log.get('response_construction', 'No construction details')}\n"
        logger.info(f"Response construction: {reasoning_log.get('response_construction', 'No construction details')}")
        
        if update_callback:
            update_callback(chain_of_thought)
            st.session_state.chain_of_thought = chain_of_thought
            
    except Exception as e:
        chain_of_thought += f"\nError during {selected_agent_name} processing: {str(e)}\n"
        logger.error(f"Error in {selected_agent_name}: {str(e)}")
        sentiment_result = {"sentiment": "NEUTRAL", "confidence": 0.5, "emotions": [], "key_points": []}
        if update_callback:
            update_callback(chain_of_thought)
            st.session_state.chain_of_thought = chain_of_thought

    # Generate recommended actions
    chain_of_thought += "\n=== Action Recommendation ===\n"
    chain_of_thought += "Generating next best actions using Groq API...\n"
    logger.info("Generating recommended actions")
    action_messages = [
        {"role": "system", "content": action_prompt.format(
            customer_data=json.dumps(customer_data),
            recent_transaction=json.dumps(recent_transaction),
            travel_notice=json.dumps(travel_notice_data),
            transcript=transcript,
            sentiment_result=json.dumps(sentiment_result)
        )}
    ]
    action_response, action_error = make_groq_request(action_messages, model, groq_api_key)
    if action_error:
        chain_of_thought += f"Action recommendation error: {action_error}\n"
        logger.error(f"Action recommendation error: {action_error}")
        recommended_actions = [{
            "action": "Follow-up Call",
            "description": "Schedule a follow-up call to address the issue manually.",
            "priority": "High",
            "category": "Customer Support",
            "icon": "📞"
        }]
    else:
        try:
            recommended_actions = json.loads(action_response)
            chain_of_thought += f"Recommended actions: {json.dumps(recommended_actions, indent=2)}\n"
            logger.info(f"Recommended actions: {recommended_actions}")
        except json.JSONDecodeError:
            chain_of_thought += "Error: Invalid JSON response from action recommendation.\n"
            logger.error("Invalid JSON response from action recommendation")
            recommended_actions = [{
                "action": "Follow-up Call",
                "description": "Schedule a follow-up call to address the issue manually.",
                "priority": "High",
                "category": "Customer Support",
                "icon": "📞"
            }]
    
    if update_callback:
        update_callback(chain_of_thought)
        st.session_state.chain_of_thought = chain_of_thought

    # Generate detailed reasoning
    chain_of_thought += "\n=== Detailed Reasoning Analysis ===\n"
    chain_of_thought += "Generating comprehensive reasoning narrative using Groq API...\n"
    logger.info("Generating detailed reasoning")
    reasoning_messages = [
        {"role": "system", "content": reasoning_prompt.format(
            customer_data=json.dumps(customer_data),
            recent_transaction=json.dumps(recent_transaction),
            travel_notice=json.dumps(travel_notice_data),
            transcript=transcript,
            sentiment_result=json.dumps(sentiment_result)
        )}
    ]
    reasoning_response, reasoning_error = make_groq_request(reasoning_messages, model, groq_api_key, max_tokens=2000)
    if reasoning_error:
        chain_of_thought += f"Reasoning analysis error: {reasoning_error}\n"
        logger.error(f"Reasoning analysis error: {reasoning_error}")
    else:
        chain_of_thought += f"Detailed reasoning:\n{reasoning_response}\n"
        logger.info(f"Detailed reasoning completed")
    
    chain_of_thought += "\n=== Analysis Complete ===\n"
    logger.info("Analysis complete")
    if update_callback:
        update_callback(chain_of_thought)
        st.session_state.chain_of_thought = chain_of_thought
    
    return sentiment_result, recommended_actions, chain_of_thought

def make_groq_request(messages, model, groq_api_key, temperature=0.7, max_tokens=1000):
    if not groq_api_key:
        logger.error("No API key provided")
        return None, "No API key provided."
    try:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",  # Use provided key
            "Content-Type": "application/json"
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        logger.debug(f"Sending Groq API request: {payload}")
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return None, f"API error: {response.status_code}. Please check your API key."
        content = response.json()["choices"][0]["message"]["content"].strip()
        logger.info("Groq API request successful")
        return content, None
    except requests.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None, f"Network error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None, f"Unexpected error: {str(e)}"

st.set_page_config(
    page_title="Next Best Action Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(STYLES, unsafe_allow_html=True)

AGENT_WELCOME_MESSAGES = {
    "TransactionAnalysisAgent": [
        "Hi! I'm here to assist with any questions about your recent transactions. What's on your mind?",
        "Hello! Need help understanding your transactions? Ask away!",
        "Got questions about your transaction history? I can help with that."
    ],
    "TravelNoticeAgent": [
        "Hello! I'm your travel assistant. Need help with travel notices or related issues?",
        "Planning a trip? I can help set up travel notices.",
        "Hi there! Let me know if you need assistance with travel plans or notices."
    ],
    "CardServicesAgent": [
        "Hi! I'm here to help with your card services. How can I assist you today?",
        "Need help with your card? Whether it's activation, replacement, or something else, I'm here.",
        "Hello! Ask me anything about your card services."
    ],
    "GeneralAgent": [
        "Hi! I'm your customer assistant. How can I help you today?",
    ]
}

session_defaults = {
    'analyzed': False,
    'customer_data': {},
    'call_transcript': "",
    'travel_notice_data': {},
    'recent_transaction': {},
    'recommended_actions': [],
    'sentiment_result': {},
    'chain_of_thought': "Click 'Run AI Analysis' to start live reasoning...",
    'api_key_set': False,
    'template_loaded': False,
    'groq_api_key': "",
    'customer_chat_history': [],
    'pending_customer_message': "",
    'last_transcript': "",
    'selected_agent': "GeneralAgent"
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if not st.session_state.customer_chat_history:
    initial_message = AGENT_WELCOME_MESSAGES.get(st.session_state.selected_agent, AGENT_WELCOME_MESSAGES["GeneralAgent"])
    st.session_state.customer_chat_history = [
        {"role": "assistant", "content": initial_message, "timestamp": datetime.now().strftime("%I:%M %p")}
    ]

if not st.session_state.template_loaded and DEMO_TEMPLATES:
    first_template_key = list(DEMO_TEMPLATES.keys())[0]
    selected_template = DEMO_TEMPLATES[first_template_key]
    st.session_state.customer_data = selected_template["customer_data"]
    st.session_state.travel_notice_data = selected_template["travel_notice_data"]
    st.session_state.recent_transaction = selected_template["recent_transaction"]
    st.session_state.template_loaded = True

st.markdown("<h1 class='main-header'>🎯 Next Best Action Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Customer Support Analysis & AI-Powered Recommendation System</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key)
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key
        st.session_state.api_key_set = True
    model_option = st.selectbox(
        "Select AI Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    st.markdown("---")

st.markdown(CHAT_BOX_STYLES, unsafe_allow_html=True)

# Build transcript
transcript = "\n".join([f"{'Customer' if msg['role'] == 'user' else 'Agent'}: {msg['content']}" 
                       for msg in st.session_state.customer_chat_history])
if st.session_state.pending_customer_message:
    transcript += f"\nCustomer: {st.session_state.pending_customer_message}"

rt = st.session_state.recent_transaction
if isinstance(rt, list):
    rt = rt[0] if rt else {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("💬 Customer Chat")
    st.markdown("Enter your customer message here and click 'Run AI Analysis' to see the reasoning process.")
    with st.container():
        for msg in st.session_state.customer_chat_history:
            role_class = "user" if msg["role"] == "user" else "assistant"
            timestamp = msg.get("timestamp", "")
            st.markdown(
                f'<div class="chat-message {role_class}">'
                f'{msg["content"]}'
                f'<div class="timestamp" style="font-size: 0.75rem; color: #888;">{timestamp}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    user_input = st.chat_input("Type message to Customer AI...", disabled=False)
    if user_input:
        st.session_state.customer_chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().strftime("%I:%M %p")})
        st.session_state.pending_customer_message = user_input
        st.rerun()

with col2:
    st.subheader("🧠 AI Chain of Thought Analysis (Live)")
    st.markdown("Detailed reasoning process, including AI-driven routing, sentiment, actions, and narrative, will appear here after running analysis:")
    # chain_output = st.session_state.chain_of_thought.replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            background-color: #f9f9f9; 
            padding: 15px; 
            border-radius: 10px; 
            border: 1px solid #ddd;
            max-height: 600px; 
            overflow-y: auto; 
            overflow-x: auto;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 14px;
            scrollbar-width: thin;
            scrollbar-color: #888 #f1f1f1;
        ">
            {st.session_state.chain_of_thought}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
if st.button("🔍 Run AI Analysis", use_container_width=True):
    if not st.session_state.api_key_set or not st.session_state.groq_api_key:
        st.error("Please enter a valid Groq API key in the sidebar.")
    elif not transcript.strip() or (not any(msg["role"] == "user" for msg in st.session_state.customer_chat_history) and not st.session_state.pending_customer_message):
        st.error("No customer input provided. Please enter a message in the Customer Chat to analyze.")
    else:
        with st.spinner("Starting live AI analysis..."):
            st.session_state.analyzed = True
            st.session_state.last_transcript = transcript

            def update_chain_of_thought(cot):
                st.session_state.chain_of_thought = cot
                placeholder.markdown(cot.replace("\n", "<br>"), unsafe_allow_html=True)

            placeholder = st.empty()
            progress_bar = st.progress(0)
            sentiment_result, recommended_actions, chain_of_thought = analyze_with_groq(
                transcript,
                st.session_state.customer_data,
                st.session_state.travel_notice_data,
                rt,
                model_option,
                st.session_state.groq_api_key,
                update_callback=update_chain_of_thought
            )
            for i in range(100):
                time.sleep(0.03)
                progress_bar.progress(i + 1)
            st.session_state.sentiment_result = sentiment_result or {}
            st.session_state.recommended_actions = recommended_actions or []
            st.session_state.chain_of_thought = chain_of_thought or "No reasoning provided."
            st.success("Analysis complete! Check the Chain of Thought for detailed AI-driven routing, sentiment, actions, and reasoning.")
            st.session_state.pending_customer_message = ""
            st.rerun()

st.header("🎯 Recommended Next Best Actions")
actions = st.session_state.recommended_actions
if actions:
    for prio in ["High", "Medium", "Low"]:
        for action in [a for a in actions if a.get("priority") == prio]:
            with st.container():
                st.markdown(f"""
                <div class="action-card">
                    <h3>{action.get('icon', '🔹')} {action.get('action', 'Action')}</h3>
                    <p>{action.get('description', '')}</p>
                    <p>Priority: <span class="priority-{action.get('priority', 'Medium').lower()}">{action.get('priority', 'Medium')}</span></p>
                    <p>Category: {action.get('category', 'General')}</p>
                </div>
                """, unsafe_allow_html=True)
else:
    st.warning("No recommended actions available.")

st.markdown("---")
col1, _, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Analysis", use_container_width=True):
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
            label="Download Analysis JSON",
            data=json.dumps(analysis_data, indent=2),
            file_name=filename,
            mime="application/json",
            use_container_width=True
        )

with col3:
    if st.button("🔄 Reset Analysis", use_container_width=True):
        for key in session_defaults.keys():
            st.session_state[key] = session_defaults[key]
        initial_message = AGENT_WELCOME_MESSAGES.get(st.session_state.selected_agent, AGENT_WELCOME_MESSAGES["GeneralAgent"])
        st.session_state.customer_chat_history = [
            {"role": "assistant", "content": initial_message, "timestamp": datetime.now().strftime("%I:%M %p")}
        ]
        st.rerun()

st.caption("Next Best Action Recommendation Engine - Enterprise v2.0")