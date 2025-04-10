import streamlit as st
import pandas as pd
import plotly.express as px
import random
import json
from datetime import datetime
import time
import logging
import requests
from constants import DEMO_TEMPLATES, sentiment_prompt, action_prompt, reasoning_prompt, STYLES, CHAT_BOX_STYLES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def analyze_with_groq(transcript, customer_data, travel_notice_data, recent_transaction, model, update_callback=None):
    sentiment_result = {}
    recommended_actions = []
    chain_of_thought = "Starting analysis...\n"

    # Sentiment Analysis
    try:
        sentiment_messages = [
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": sentiment_prompt.format(transcript=transcript)}
        ]
        logger.info(">>> Sentiment analysis started")
        sentiment_result_raw, error = make_groq_request(sentiment_messages, model)
        if error:
            chain_of_thought += f"Sentiment analysis failed: {error}\n"
            sentiment_result = {"sentiment": "NEUTRAL", "confidence": 0.5, "emotions": [], "key_points": []}
        else:
            try:
                if isinstance(sentiment_result_raw, str) and sentiment_result_raw.startswith("{") and sentiment_result_raw.endswith("}"):
                    sentiment_result = json.loads(sentiment_result_raw.strip())
                else:
                    sentiment_result = {"sentiment": "NEUTRAL", "confidence": 0.5, "emotions": [], "key_points": []}
                chain_of_thought += "Sentiment analysis completed.\n"
            except (json.JSONDecodeError, TypeError):
                sentiment_result = {"sentiment": "NEUTRAL", "confidence": 0.5, "emotions": [], "key_points": []}
                chain_of_thought += "Error parsing sentiment result, defaulting to NEUTRAL.\n"
        if update_callback:
            update_callback(chain_of_thought)
    except Exception as e:
        chain_of_thought += f"Sentiment analysis error: {str(e)}\n"
        sentiment_result = {"sentiment": "NEUTRAL", "confidence": 0.5, "emotions": [], "key_points": []}
        if update_callback:
            update_callback(chain_of_thought)

    # Action Recommendation
    try:
        action_input = action_prompt.format(
            customer_data=json.dumps(customer_data),
            transcript=transcript,
            travel_notice=json.dumps(travel_notice_data),
            recent_transaction=json.dumps(recent_transaction),
            sentiment_result=json.dumps(sentiment_result)
        )
        action_messages = [
            {"role": "system", "content": "You are a customer support assistant recommending next best actions."},
            {"role": "user", "content": action_input}
        ]
        actions_result, error = make_groq_request(action_messages, model)
        if error:
            chain_of_thought += f"Action recommendation failed: {error}\n"
            recommended_actions = [{"action": "Follow-up Call", "description": "Schedule a follow-up call.", "priority": "High", "category": "Customer Support"}]
        else:
            try:
                recommended_actions = json.loads(actions_result) if actions_result else [
                    {"action": "Follow-up Call", "description": "Schedule a follow-up call.", "priority": "High", "category": "Customer Support"}
                ]
                chain_of_thought += "Action recommendations generated.\n"
            except json.JSONDecodeError:
                recommended_actions = [{"action": "Follow-up Call", "description": "Schedule a follow-up call.", "priority": "High", "category": "Customer Support"}]
                chain_of_thought += "Error parsing actions, using default recommendation.\n"
        if update_callback:
            update_callback(chain_of_thought)
    except Exception as e:
        chain_of_thought += f"Action recommendation error: {str(e)}\n"
        recommended_actions = [{"action": "Follow-up Call", "description": "Schedule a follow-up call.", "priority": "High", "category": "Customer Support"}]
        if update_callback:
            update_callback(chain_of_thought)

    # Reasoning (Chain of Thought)
    try:
        reasoning_input = reasoning_prompt.format(
            customer_data=json.dumps(customer_data),
            transcript=transcript,
            travel_notice=json.dumps(travel_notice_data),
            recent_transaction=json.dumps(recent_transaction),
            sentiment_result=json.dumps(sentiment_result)
        )
        reasoning_messages = [
            {"role": "system", "content": "You are an assistant explaining your reasoning."},
            {"role": "user", "content": reasoning_input}
        ]
        reasoning_result, error = make_groq_request(reasoning_messages, model)
        if error:
            chain_of_thought += f"Reasoning failed: {error}\n"
        else:
            chain_of_thought += f"Reasoning: {reasoning_result}\n"
        if update_callback:
            update_callback(chain_of_thought)
    except Exception as e:
        chain_of_thought += f"Reasoning error: {str(e)}\n"
        if update_callback:
            update_callback(chain_of_thought)

    return sentiment_result, recommended_actions, chain_of_thought

st.set_page_config(
    page_title="Next Best Action Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(STYLES, unsafe_allow_html=True)

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
    'customer_chat_history': [{"role": "assistant", "content": "Hi! I'm your customer assistant. How can I help you today?"}],
    'support_chat_history': [{"role": "assistant", "content": "Hi! I'm Lisa AI, your support assistant. How can I assist you?"}],
    'pending_customer_message': "",
    'lisa_response': "",
    'awaiting_transfer': False,
    'use_lisa_for_customer': False,
    'show_response_preview': True,
    'last_transcript': ""
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

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

def make_groq_request(messages, model, temperature=0.7, max_tokens=1000):
    if not st.session_state.groq_api_key:
        return None, "No API key provided. Please enter your Groq API key in the sidebar."
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.groq_api_key}",
            "Content-Type": "application/json"
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return None, f"API error: {response.status_code}. Please check your API key and try again."
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content, None
    except requests.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None, f"Network error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None, f"Unexpected error: {str(e)}"

st.markdown(CHAT_BOX_STYLES, unsafe_allow_html=True)

if st.session_state.use_lisa_for_customer and st.session_state.lisa_response:
    st.session_state.customer_chat_history.append({"role": "assistant", "content": st.session_state.lisa_response, "timestamp": datetime.now().strftime("%I:%M %p")})
    st.session_state.use_lisa_for_customer = False
    st.session_state.lisa_response = ""
    st.session_state.awaiting_transfer = False
    st.session_state.show_response_preview = False

transcript = "\n".join([f"{'Customer' if msg['role'] == 'user' else 'Agent'}: {msg['content']}" 
                       for msg in st.session_state.customer_chat_history])
rt = st.session_state.recent_transaction
if isinstance(rt, list):
    rt = rt[0] if rt else {}

if st.session_state.analyzed and st.session_state.last_transcript != transcript and st.session_state.api_key_set:
    with st.spinner("Updating AI reasoning..."):
        sentiment_result, recommended_actions, chain_of_thought = analyze_with_groq(
            transcript,
            st.session_state.customer_data,
            st.session_state.travel_notice_data,
            rt,
            model_option,
            lambda cot: st.session_state.update({'chain_of_thought': cot})
        )
        st.session_state.sentiment_result = sentiment_result or {}
        st.session_state.recommended_actions = recommended_actions or []
        st.session_state.chain_of_thought = chain_of_thought or "No reasoning provided."
        st.session_state.last_transcript = transcript

col1, col2 = st.columns(2)

with col1:
    st.subheader("💬 Customer Chat")
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
    user_input = st.chat_input("Type message to Customer AI...")
    if user_input:
        st.session_state.customer_chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().strftime("%I:%M %p")})
        st.session_state.pending_customer_message = user_input
        st.session_state.awaiting_transfer = True
        st.session_state.show_response_preview = True
        st.rerun()

with col2:
    if not st.session_state.analyzed:
        st.subheader("🤖 Lisa AI Assistant")
        st.markdown("""
            <h6 style='font-size: 0.85rem; color: #555;'>
            Hi Prabhu, I am Lisa AI, your AI Assistant. I am here to assist you with crafting meaningful responses that align with your customers' needs and preferences.
            </h6>
            """, unsafe_allow_html=True)
        with st.container():
            for msg in st.session_state.support_chat_history:
                role_class = "user" if msg["role"] == "user" else "assistant"
                st.markdown(f'<div class="chat-message {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
        if st.session_state.awaiting_transfer and st.session_state.pending_customer_message and st.session_state.show_response_preview:
            with st.spinner("Lisa AI is analyzing the customer query..."):
                prompt = f"Please suggest a response for this customer query: '{st.session_state.pending_customer_message}'"
                lisa_messages = [
                    {"role": "system", "content": "You are Lisa AI, an empathetic support assistant for helping sales reps craft messages."},
                    {"role": "user", "content": prompt}
                ]
                lisa_response, err = make_groq_request(lisa_messages, model_option)
                st.session_state.lisa_response = lisa_response or err
            st.markdown("### Suggested Response:")
            st.markdown(f'<div class="response-preview">{st.session_state.lisa_response}</div>', unsafe_allow_html=True)
            if st.button("✅ Send This Response", use_container_width=True):
                st.session_state.support_chat_history.append({"role": "user", "content": f"Customer asked: {st.session_state.pending_customer_message}"})
                st.session_state.use_lisa_for_customer = True
                st.session_state.show_response_preview = False
                st.rerun()
            if st.button("🔄 Re-generate Another Response", use_container_width=True):
                st.session_state.support_chat_history.append({"role": "user", "content": f"Customer asked: {st.session_state.pending_customer_message}"})
                with st.spinner("Lisa AI is generating a new response..."):
                    prompt = f"Please suggest a different response for this customer query: '{st.session_state.pending_customer_message}'"
                    lisa_messages = [
                        {"role": "system", "content": "You are Lisa AI, an empathetic support assistant for helping sales reps craft messages. Generate a different response than previously provided."},
                        {"role": "user", "content": prompt}
                    ]
                    lisa_response, err = make_groq_request(lisa_messages, model_option, temperature=0.8)
                    st.session_state.lisa_response = lisa_response or err
                st.session_state.show_response_preview = True
                st.rerun()
        if not st.session_state.awaiting_transfer:
            support_input = st.chat_input("Type message to Lisa AI...", key="lisa_input")
            if support_input:
                st.session_state.support_chat_history.append({"role": "user", "content": support_input})
                with st.spinner("Lisa AI is typing..."):
                    reply, err = make_groq_request([
                        {"role": "system", "content": "You are Lisa AI, an empathetic support assistant for helping sales reps craft messages."},
                        *st.session_state.support_chat_history[-5:]
                    ], model_option)
                    st.session_state.support_chat_history.append({"role": "assistant", "content": reply or err})
                st.rerun()
        if st.session_state.awaiting_transfer and not st.session_state.show_response_preview:
            st.markdown("### Write Custom Response:")
            custom_response = st.text_area("Your response to the customer:", height=100)
            if st.button("✅ Send Custom Response", use_container_width=True):
                st.session_state.support_chat_history.append({"role": "user", "content": f"Customer asked: {st.session_state.pending_customer_message}"})
                st.session_state.customer_chat_history.append({"role": "assistant", "content": custom_response, "timestamp": datetime.now().strftime("%I:%M %p")})
                st.session_state.pending_customer_message = ""
                st.session_state.awaiting_transfer = False
                st.session_state.show_response_preview = False
                st.rerun()
    else:
        st.subheader("🧠 AI Chain of Thought Analysis (Live)")
        st.markdown(st.session_state.chain_of_thought)

st.markdown("---")
if st.button("🔍 Run AI Analysis", use_container_width=True):
    if not st.session_state.api_key_set or not st.session_state.groq_api_key:
        st.error("Please enter a valid Groq API key in the sidebar.")
    else:
        with st.spinner("Starting live AI analysis..."):
            st.session_state.analyzed = True
            st.session_state.last_transcript = transcript

            def update_chain_of_thought(cot):
                st.session_state.chain_of_thought = cot
                # Force a lightweight rerender by updating a placeholder
                placeholder.markdown(cot)

            placeholder = st.empty()  # Placeholder to update chain of thought live
            progress_bar = st.progress(0)
            sentiment_result, recommended_actions, chain_of_thought = analyze_with_groq(
                transcript,
                st.session_state.customer_data,
                st.session_state.travel_notice_data,
                rt,
                model_option,
                update_chain_of_thought
            )
            for i in range(100):
                time.sleep(0.03)  # Adjusted for smoother visual effect
                progress_bar.progress(i + 1)
            st.session_state.sentiment_result = sentiment_result or {}
            st.session_state.recommended_actions = recommended_actions or []
            st.session_state.chain_of_thought = chain_of_thought or "No reasoning provided."
            st.success("Live analysis started! Chain of Thought will update as the chat evolves.")
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
        for key in ["analyzed", "sentiment_result", "recommended_actions", "chain_of_thought", "last_transcript"]:
            st.session_state[key] = session_defaults[key]
        st.rerun()

st.caption("Next Best Action Recommendation Engine - Enterprise v2.0")