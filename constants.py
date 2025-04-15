STYLES = """
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
"""
CHAT_BOX_STYLES = """
<style>
.chat-message {
    margin-bottom: 1rem;
    padding: 0.8rem;
    border-radius: 10px;
    max-width: 60%;
    margin-bottom: 1rem;
}
.user {
    background-color: #dcf8c6;
    margin-left: auto;
    max-width: 60%;
}
.assistant {
    background-color: #ffffff;
    border: 1px solid #ddd;
}
.timestamp {
    text-align: right;
    margin-top: 0.25rem;
</style>
"""

DEMO_TEMPLATES = {
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
        "travel_notice_data": {
            "submitted_date": "May 2, 2023",
            "travel_start": "May 5, 2023",
            "travel_end": "May 15, 2023",
            "countries": ["France", "Italy", "Spain"],
            "status": "Submitted but not activated due to system error",
            "submission_channel": "Mobile App"
        },
    "recent_transaction": [
    {
        "date": "April 4, 2025",
        "merchant": "Starbucks",
        "location": "New York, USA",
        "amount": "$5.75",
        "status": "Approved",
        "card_used": "World Traveler Visa ending in 7842"
    },
    {
        "date": "April 2, 2025",
        "merchant": "Tokyo Central Market",
        "location": "Tokyo, Japan",
        "amount": "¥3,200",
        "status": "Declined",
        "reason": "Insufficient funds",
        "card_used": "World Traveler Visa ending in 7842"
    },
    {
        "date": "March 30, 2025",
        "merchant": "Uber",
        "location": "Berlin, Germany",
        "amount": "€22.40",
        "status": "Approved",
        "card_used": "World Traveler Visa ending in 7842"
    },
    {
        "date": "March 25, 2025",
        "merchant": "La Casa Tapas",
        "location": "Barcelona, Spain",
        "amount": "€65.00",
        "status": "Declined",
        "reason": "Card reported lost",
        "card_used": "World Traveler Visa ending in 7842"
    },
    {
        "date": "March 22, 2025",
        "merchant": "Amazon Marketplace",
        "location": "Online",
        "amount": "$120.99",
        "status": "Approved",
        "card_used": "World Traveler Visa ending in 7842"
    }
]

    }
}

sentiment_prompt = """
You are a sentiment analysis engine.

Given a customer message, respond with a **strict JSON object** in the following format:

{{
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "confidence": float between 0 and 1,
  "emotions": [list of detected emotions like "joy", "anger", "frustration", etc.],
  "key_points": [list of important phrases from the input]
}}

Return ONLY the JSON. Do NOT include explanations, markdown formatting, or any extra text.

INPUT:
"{transcript}"
"""


action_prompt = """
You are a next-best-action generator for virtual banking assistants.

Using the customer interaction data below, generate 5 recommended actions in JSON array format.

Each action must contain:
- "action": A short, actionable title
- "description": Clear, step-by-step guidance for the agent
- "priority": One of ["High", "Medium", "Low"]
- "category": One of the following:
    - "Technical Resolution" (e.g., fixing system or card issues)
    - "Customer Service" (e.g., updating info, sending replacements)
    - "Sales Opportunity" (e.g., upsell, upgrade suggestions)
    - "Fraud Prevention" (e.g., flag unusual activity)
    - "General Inquiry" (e.g., questions not requiring action)

DATA:
- CUSTOMER INFO: {customer_data}
- RECENT TRANSACTION: {recent_transaction}
- TRAVEL NOTICE: {travel_notice}
- CALL TRANSCRIPT: {transcript}
- SENTIMENT ANALYSIS: {sentiment_result}

Only return clean JSON. No markdown. No explanation. Be specific and avoid generic actions.
"""



reasoning_prompt = """
You are a senior customer experience strategist at a global bank. Perform a diagnostic breakdown of the interaction below using expert-level reasoning.

Respond with detailed, structured text under each of these headers:

1. **Customer Context Analysis** — Based only on the data, who is this customer? What matters to them?
2. **Problem Identification** — What is the core issue or complaint?
3. **Emotional Impact Assessment** — How is the customer emotionally affected? What specific phrases reflect this?
4. **Priority Determination** — Rate the urgency (High, Medium, Low) and justify based on risk, impact, or sentiment.
5. **Opportunity Analysis** — Are there meaningful upsell, loyalty, or recovery opportunities?
6. **Long-term Relationship Considerations** — What specific actions will increase trust and long-term retention?

DATA PROVIDED:
- CUSTOMER INFO: {customer_data}
- RECENT TRANSACTION: {recent_transaction}
- TRAVEL NOTICE: {travel_notice}
- CALL TRANSCRIPT: {transcript}
- SENTIMENT ANALYSIS: {sentiment_result}

Stick to the data. No guessing or padding.
"""

