# constants.py

# CSS Styles
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

# Demo templates
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