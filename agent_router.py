# import re
# import json
# from typing import Dict, List, Tuple, Any, Optional
# import requests
# import logging

# logger = logging.getLogger(__name__)

# class RouterAgent:
#     """
#     AI-driven agent responsible for analyzing user prompts and routing to specialized agents using Groq API.
#     Generates a detailed reasoning_log explaining its routing decision.
#     """

#     def __init__(self, customer_data: Dict, travel_notice_data: Dict, recent_transactions: List[Dict], groq_api_key: str, model: str = "llama3-8b-8192"):
#         self.customer_data = customer_data
#         self.travel_notice_data = travel_notice_data
#         self.recent_transactions = recent_transactions
#         self.groq_api_key = "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"
#         self.model = model
#         self.agents = [
#             "TravelNoticeAgent",
#             "TransactionAnalysisAgent",
#             "CardServicesAgent",
#             "GeneralInquiryAgent"
#         ]
#         self.routing_prompt = """
# You are an intelligent routing assistant for a banking customer service system. Your task is to analyze a customer query and select the most appropriate specialized agent to handle it from the following options: {agents}.

# **Context Data**:
# - Customer Info: {customer_data}
# - Recent Transactions: {recent_transactions}
# - Travel Notices: {travel_notice_data}
# - Customer Query: {query}

# **Agent Descriptions**:
# - **TravelNoticeAgent**: Handles queries about travel plans, international transactions, or travel notifications (e.g., "I'm traveling to Paris", "card declined abroad").
# - **TransactionAnalysisAgent**: Handles queries about specific transactions, payments, or declines (e.g., "Why was my payment declined?", "Check my recent purchase").
# - **CardServicesAgent**: Handles queries about card issues, such as lost/stolen cards, card activation, or credit limits (e.g., "I lost my card", "Increase my credit limit").
# - **GeneralInquiryAgent**: Handles general questions, account inquiries, or unspecified issues (e.g., "How do I check my balance?", "I need help").

# **Instructions**:
# 1. Analyze the query and context to determine the best agent.
# 2. Return a JSON object with:
#    - "agent": The selected agent name (one of {agents}).
#    - "reasoning": A detailed explanation of why this agent was chosen, including relevant keywords, context clues, and query intent.
#    - "confidence": A float between 0 and 1 indicating confidence in the decision.
# Return ONLY the JSON object, no extra text.
#         """

#     def route(self, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
#         """
#         Analyzes the user prompt using Groq API and routes to the appropriate agent.
#         Returns: (agent_name, reasoning_log)
#         """
#         reasoning_log = {
#             "input_prompt": user_prompt,
#             "keyword_matches": {},
#             "pattern_matches": {},
#             "context_analysis": {},
#             "routing_decision": "",
#             "confidence_scores": {},
#             "final_agent": ""
#         }

#         # Perform rule-based analysis for transparency (optional, to enrich reasoning_log)
#         self._rule_based_analysis(user_prompt, reasoning_log)

#         # Prepare Groq API call
#         prompt = self.routing_prompt.format(
#             agents=self.agents,
#             customer_data=json.dumps(self.customer_data, indent=2),
#             recent_transactions=json.dumps(self.recent_transactions, indent=2),
#             travel_notice_data=json.dumps(self.travel_notice_data, indent=2),
#             query=user_prompt
#         )
#         messages = [{"role": "system", "content": prompt}]

#         # Make Groq API call
#         response, error = self._make_groq_request(messages)
#         if error:
#             logger.error(f"Groq API error: {error}")
#             reasoning_log["routing_decision"] = f"Error in AI routing: {error}. Defaulting to GeneralInquiryAgent."
#             reasoning_log["final_agent"] = "GeneralInquiryAgent"
#             reasoning_log["confidence_scores"] = {agent: 0.0 for agent in self.agents}
#             reasoning_log["confidence_scores"]["GeneralInquiryAgent"] = 0.5
#             return "GeneralInquiryAgent", reasoning_log

#         # Parse AI response
#         try:
#             result = json.loads(response)
#             selected_agent = result.get("agent", "GeneralInquiryAgent")
#             ai_reasoning = result.get("reasoning", "No reasoning provided by AI.")
#             confidence = result.get("confidence", 0.5)

#             # Validate selected agent
#             if selected_agent not in self.agents:
#                 selected_agent = "GeneralInquiryAgent"
#                 ai_reasoning = f"Invalid agent selected by AI. Defaulting to GeneralInquiryAgent. Original reasoning: {ai_reasoning}"
#                 confidence = 0.5

#             # Update reasoning_log
#             reasoning_log["routing_decision"] = f"AI selected {selected_agent}. Reasoning: {ai_reasoning}"
#             reasoning_log["final_agent"] = selected_agent
#             reasoning_log["confidence_scores"] = {agent: 0.0 for agent in self.agents}
#             reasoning_log["confidence_scores"][selected_agent] = confidence
#             reasoning_log["ai_reasoning"] = ai_reasoning

#         except json.JSONDecodeError:
#             logger.error("Invalid JSON response from Groq API")
#             reasoning_log["routing_decision"] = "Error parsing AI response. Defaulting to GeneralInquiryAgent."
#             reasoning_log["final_agent"] = "GeneralInquiryAgent"
#             reasoning_log["confidence_scores"] = {agent: 0.0 for agent in self.agents}
#             reasoning_log["confidence_scores"]["GeneralInquiryAgent"] = 0.5
#             selected_agent = "GeneralInquiryAgent"

#         return selected_agent, reasoning_log

#     def _rule_based_analysis(self, user_prompt: str, reasoning_log: Dict[str, Any]):
#         """
#         Performs rule-based analysis to enrich reasoning_log (optional, for transparency).
#         """
#         routing_rules = [
#             {
#                 "agent": "TravelNoticeAgent",
#                 "keywords": ["travel notice", "travel plan", "trip notification", "going abroad", "traveling to",
#                             "activate travel", "travel alert", "international travel", "foreign transaction"],
#                 "patterns": [
#                     r"(?i).*\b(travel|trip)\b.*\b(notice|notification|alert)\b.*",
#                     r"(?i).*\b(activate|update|submit)\b.*\b(travel|trip)\b.*",
#                     r"(?i).*\b(travel|trip|traveling|going)\b.*\b(to|abroad|overseas|internationally)\b.*"
#                 ]
#             },
#             {
#                 "agent": "TransactionAnalysisAgent",
#                 "keywords": ["transaction", "purchase", "payment", "declined", "approved", "charge",
#                             "spent", "buy", "bought", "paid", "decline"],
#                 "patterns": [
#                     r"(?i).*\b(transaction|purchase|payment|charge)\b.*\b(declined|denied|failed|rejected)\b.*",
#                     r"(?i).*\b(why|how)\b.*\b(transaction|payment|card)\b.*\b(declined|denied|failed|rejected)\b.*",
#                     r"(?i).*\b(check|review|view|explain)\b.*\b(transaction|purchase|payment|charge)\b.*"
#                 ]
#             },
#             {
#                 "agent": "CardServicesAgent",
#                 "keywords": ["card", "credit card", "debit card", "visa", "mastercard", "replace", "activate card",
#                             "lost card", "stolen card", "new card", "card limit", "credit limit"],
#                 "patterns": [
#                     r"(?i).*\b(card)\b.*\b(lost|stolen|damaged|broken|replace|new|activate)\b.*",
#                     r"(?i).*\b(credit|debit)\b.*\b(limit|balance|available|increase|decrease)\b.*",
#                     r"(?i).*\b(report|freeze|block|unblock|lock|unlock)\b.*\b(card|account)\b.*"
#                 ]
#             },
#             {
#                 "agent": "GeneralInquiryAgent",
#                 "keywords": ["help", "support", "question", "inquiry", "information", "how do I", "how to",
#                             "what is", "account", "balance", "statement"],
#                 "patterns": [
#                     r"(?i).*\b(what|how|when|where|why|who)\b.*\b(account|balance|statement|fee|charge)\b.*",
#                     r"(?i).*\b(help|assist|support)\b.*\b(with|me|please|need)\b.*",
#                     r"(?i).*\b(account|profile|settings|preferences)\b.*\b(view|change|update|modify)\b.*"
#                 ]
#             }
#         ]

#         # Keyword matches
#         for rule in routing_rules:
#             agent = rule["agent"]
#             keyword_matches = [kw for kw in rule["keywords"] if kw.lower() in user_prompt.lower()]
#             if keyword_matches:
#                 reasoning_log["keyword_matches"][agent] = keyword_matches

#         # Pattern matches
#         for rule in routing_rules:
#             agent = rule["agent"]
#             pattern_matches = [p for p in rule["patterns"] if re.search(p, user_prompt)]
#             if pattern_matches:
#                 reasoning_log["pattern_matches"][agent] = pattern_matches

#         # Context analysis
#         context_clues = self._analyze_context(user_prompt)
#         reasoning_log["context_analysis"] = context_clues

#     def _analyze_context(self, user_prompt: str) -> Dict[str, int]:
#         """Analyzes user prompt against recent activity for additional context."""
#         context_clues = {}
#         prompt_lower = user_prompt.lower()

#         # Check for mentions of recent transactions
#         for transaction in self.recent_transactions:
#             merchant = transaction.get("merchant", "").lower()
#             location = transaction.get("location", "").lower()
#             status = transaction.get("status", "").lower()
#             if re.search(r'\b' + re.escape(merchant) + r'\b', prompt_lower) or \
#                re.search(r'\b' + re.escape(location) + r'\b', prompt_lower):
#                 base_agent = "TransactionAnalysisAgent"
#                 current_score = context_clues.get(base_agent, 0)
#                 context_clues[base_agent] = max(current_score, 2 if status == "declined" else 1)

#         # Check for travel notice related context
#         for country in self.travel_notice_data.get("countries", []):
#             if re.search(r'\b' + re.escape(country.lower()) + r'\b', prompt_lower):
#                 context_clues["TravelNoticeAgent"] = context_clues.get("TravelNoticeAgent", 0) + 2

#         if re.search(r"\btravel\b", prompt_lower) and re.search(r"\bnotice\b", prompt_lower):
#             context_clues["TravelNoticeAgent"] = context_clues.get("TravelNoticeAgent", 0) + 3

#         # Check for card-related context
#         if re.search(r"\blost\b", prompt_lower) and re.search(r"\bcard\b", prompt_lower):
#             context_clues["CardServicesAgent"] = context_clues.get("CardServicesAgent", 0) + 3

#         # Check for specific location keywords hinting at travel
#         travel_locations = ["tokyo", "japan", "berlin", "germany", "barcelona", "spain"]
#         if any(re.search(r'\b' + loc + r'\b', prompt_lower) for loc in travel_locations):
#             context_clues["TravelNoticeAgent"] = context_clues.get("TravelNoticeAgent", 0) + 1

#         return context_clues

#     def _make_groq_request(self, messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
#         """Makes a request to the Groq API."""
#         try:
#             headers = {
#                 "Authorization": f"Bearer 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",
#                 "Content-Type": "application/json"
#             }
#             url = "https://api.groq.com/openai/v1/chat/completions"
#             payload = {
#                 "model": self.model,
#                 "messages": messages,
#                 "temperature": 0.7,
#                 "max_tokens": 1000
#             }
#             response = requests.post(url, headers=headers, json=payload)
#             if response.status_code != 200:
#                 return None, f"API error: {response.status_code} - {response.text}"
#             content = response.json()["choices"][0]["message"]["content"].strip()
#             return content, None
#         except requests.RequestException as e:
#             return None, f"Network error: {str(e)}"
#         except Exception as e:
#             return None, f"Unexpected error: {str(e)}"

import re
import json
from typing import Dict, List, Tuple, Any, Optional
import requests
import logging

# Use the same logger as sai.py
logger = logging.getLogger('ChainOfThought')

class RouterAgent:
    """
    AI-driven agent responsible for analyzing user prompts and routing to specialized agents using Groq API.
    Generates a detailed reasoning_log explaining its routing decision.
    """

    def __init__(self, customer_data: Dict, travel_notice_data: Dict, recent_transactions: List[Dict], groq_api_key: str, model: str = "llama3-8b-8192"):
        self.customer_data = customer_data
        self.travel_notice_data = travel_notice_data
        self.recent_transactions = recent_transactions
        self.groq_api_key = groq_api_key  # Use provided API key
        self.model = model
        self.agents = [
            "TravelNoticeAgent",
            "TransactionAnalysisAgent",
            "CardServicesAgent",
            "GeneralInquiryAgent"
        ]
        self.routing_prompt = """
You are an intelligent routing assistant for a banking customer service system. Your task is to analyze a customer query and select the most appropriate specialized agent to handle it from the following options: {agents}.

**Context Data**:
- Customer Info: {customer_data}
- Recent Transactions: {recent_transactions}
- Travel Notices: {travel_notice_data}
- Customer Query: {query}

**Agent Descriptions**:
- **TravelNoticeAgent**: Handles queries about travel plans, international transactions, or travel notifications (e.g., "I'm traveling to Paris", "card declined abroad").
- **TransactionAnalysisAgent**: Handles queries about specific transactions, payments, or declines (e.g., "Why was my payment declined?", "Check my recent purchase").
- **CardServicesAgent**: Handles queries about card issues, such as lost/stolen cards, card activation, or credit limits (e.g., "I lost my card", "Increase my credit limit").
- **GeneralInquiryAgent**: Handles general questions, account inquiries, or unspecified issues (e.g., "How do I check my balance?", "I need help").

**Instructions**:
1. Analyze the query and context to determine the best agent.
2. Return a JSON object with:
   - "agent": The selected agent name (one of {agents}).
   - "reasoning": A detailed explanation of why this agent was chosen, including relevant keywords, context clues, and query intent.
   - "confidence": A float between 0 and 1 indicating confidence in the decision.
Return ONLY the JSON object, no extra text.
        """
        logger.info("RouterAgent initialized with model: %s", model)

    def route(self, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyzes the user prompt using Groq API and routes to the appropriate agent.
        Returns: (agent_name, reasoning_log)
        """
        logger.info("Routing user prompt: %s", user_prompt)
        reasoning_log = {
            "input_prompt": user_prompt,
            "keyword_matches": {},
            "pattern_matches": {},
            "context_analysis": {},
            "routing_decision": "",
            "confidence_scores": {},
            "final_agent": ""
        }

        # Perform rule-based analysis for transparency
        logger.debug("Performing rule-based analysis")
        self._rule_based_analysis(user_prompt, reasoning_log)

        # Prepare Groq API call
        prompt = self.routing_prompt.format(
            agents=self.agents,
            customer_data=json.dumps(self.customer_data, indent=2),
            recent_transactions=json.dumps(self.recent_transactions, indent=2),
            travel_notice_data=json.dumps(self.travel_notice_data, indent=2),
            query=user_prompt
        )
        messages = [{"role": "system", "content": prompt}]
        logger.debug("Prepared Groq API prompt with %d messages", len(messages))

        # Make Groq API call
        response, error = self._make_groq_request(messages)
        if error:
            logger.error("Groq API error: %s", error)
            reasoning_log["routing_decision"] = f"Error in AI routing: {error}. Defaulting to GeneralInquiryAgent."
            reasoning_log["final_agent"] = "GeneralInquiryAgent"
            reasoning_log["confidence_scores"] = {agent: 0.0 for agent in self.agents}
            reasoning_log["confidence_scores"]["GeneralInquiryAgent"] = 0.5
            logger.info("Defaulted to GeneralInquiryAgent due to API error")
            return "GeneralInquiryAgent", reasoning_log

        # Parse AI response
        try:
            result = json.loads(response)
            selected_agent = result.get("agent", "GeneralInquiryAgent")
            ai_reasoning = result.get("reasoning", "No reasoning provided by AI.")
            confidence = result.get("confidence", 0.5)

            # Validate selected agent
            if selected_agent not in self.agents:
                logger.warning("Invalid agent selected: %s. Defaulting to GeneralInquiryAgent", selected_agent)
                selected_agent = "GeneralInquiryAgent"
                ai_reasoning = f"Invalid agent selected by AI. Defaulting to GeneralInquiryAgent. Original reasoning: {ai_reasoning}"
                confidence = 0.5

            # Update reasoning_log
            reasoning_log["routing_decision"] = f"AI selected {selected_agent}. Reasoning: {ai_reasoning}"
            reasoning_log["final_agent"] = selected_agent
            reasoning_log["confidence_scores"] = {agent: 0.0 for agent in self.agents}
            reasoning_log["confidence_scores"][selected_agent] = confidence
            reasoning_log["ai_reasoning"] = ai_reasoning
            logger.info("Selected agent: %s with confidence: %.2f", selected_agent, confidence)
            logger.debug("Routing log: %s", reasoning_log)

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON response from Groq API: %s", str(e))
            reasoning_log["routing_decision"] = "Error parsing AI response. Defaulting to GeneralInquiryAgent."
            reasoning_log["final_agent"] = "GeneralInquiryAgent"
            reasoning_log["confidence_scores"] = {agent: 0.0 for agent in self.agents}
            reasoning_log["confidence_scores"]["GeneralInquiryAgent"] = 0.5
            logger.info("Defaulted to GeneralInquiryAgent due to JSON error")
            selected_agent = "GeneralInquiryAgent"

        return selected_agent, reasoning_log

    def _rule_based_analysis(self, user_prompt: str, reasoning_log: Dict[str, Any]):
        """
        Performs rule-based analysis to enrich reasoning_log (optional, for transparency).
        """
        logger.debug("Starting rule-based analysis for prompt: %s", user_prompt)
        routing_rules = [
            {
                "agent": "TravelNoticeAgent",
                "keywords": ["travel notice", "travel plan", "trip notification", "going abroad", "traveling to",
                            "activate travel", "travel alert", "international travel", "foreign transaction"],
                "patterns": [
                    r"(?i).*\b(travel|trip)\b.*\b(notice|notification|alert)\b.*",
                    r"(?i).*\b(activate|update|submit)\b.*\b(travel|trip)\b.*",
                    r"(?i).*\b(travel|trip|traveling|going)\b.*\b(to|abroad|overseas|internationally)\b.*"
                ]
            },
            {
                "agent": "TransactionAnalysisAgent",
                "keywords": ["transaction", "purchase", "payment", "declined", "approved", "charge",
                            "spent", "buy", "bought", "paid", "decline"],
                "patterns": [
                    r"(?i).*\b(transaction|purchase|payment|charge)\b.*\b(declined|denied|failed|rejected)\b.*",
                    r"(?i).*\b(why|how)\b.*\b(transaction|payment|card)\b.*\b(declined|denied|failed|rejected)\b.*",
                    r"(?i).*\b(check|review|view|explain)\b.*\b(transaction|purchase|payment|charge)\b.*"
                ]
            },
            {
                "agent": "CardServicesAgent",
                "keywords": ["card", "credit card", "debit card", "visa", "mastercard", "replace", "activate card",
                            "lost card", "stolen card", "new card", "card limit", "credit limit"],
                "patterns": [
                    r"(?i).*\b(card)\b.*\b(lost|stolen|damaged|broken|replace|new|activate)\b.*",
                    r"(?i).*\b(credit|debit)\b.*\b(limit|balance|available|increase|decrease)\b.*",
                    r"(?i).*\b(report|freeze|block|unblock|lock|unlock)\b.*\b(card|account)\b.*"
                ]
            },
            {
                "agent": "GeneralInquiryAgent",
                "keywords": ["help", "support", "question", "inquiry", "information", "how do I", "how to",
                            "what is", "account", "balance", "statement"],
                "patterns": [
                    r"(?i).*\b(what|how|when|where|why|who)\b.*\b(account|balance|statement|fee|charge)\b.*",
                    r"(?i).*\b(help|assist|support)\b.*\b(with|me|please|need)\b.*",
                    r"(?i).*\b(account|profile|settings|preferences)\b.*\b(view|change|update|modify)\b.*"
                ]
            }
        ]

        # Keyword matches
        for rule in routing_rules:
            agent = rule["agent"]
            keyword_matches = [kw for kw in rule["keywords"] if kw.lower() in user_prompt.lower()]
            if keyword_matches:
                reasoning_log["keyword_matches"][agent] = keyword_matches
                logger.debug("Keyword matches for %s: %s", agent, keyword_matches)

        # Pattern matches
        for rule in routing_rules:
            agent = rule["agent"]
            pattern_matches = [p for p in rule["patterns"] if re.search(p, user_prompt)]
            if pattern_matches:
                reasoning_log["pattern_matches"][agent] = pattern_matches
                logger.debug("Pattern matches for %s: %s", agent, pattern_matches)

        # Context analysis
        logger.debug("Performing context analysis")
        context_clues = self._analyze_context(user_prompt)
        reasoning_log["context_analysis"] = context_clues
        if context_clues:
            logger.debug("Context analysis results: %s", context_clues)
        else:
            logger.debug("No context clues found")

    def _analyze_context(self, user_prompt: str) -> Dict[str, int]:
        """Analyzes user prompt against recent activity for additional context."""
        logger.debug("Analyzing context for prompt: %s", user_prompt)
        context_clues = {}
        prompt_lower = user_prompt.lower()

        # Check for mentions of recent transactions
        for transaction in self.recent_transactions:
            merchant = transaction.get("merchant", "").lower()
            location = transaction.get("location", "").lower()
            status = transaction.get("status", "").lower()
            if re.search(r'\b' + re.escape(merchant) + r'\b', prompt_lower) or \
               re.search(r'\b' + re.escape(location) + r'\b', prompt_lower):
                base_agent = "TransactionAnalysisAgent"
                current_score = context_clues.get(base_agent, 0)
                context_clues[base_agent] = max(current_score, 2 if status == "declined" else 1)
                logger.debug("Transaction context matched for %s: merchant=%s, location=%s, score=%d", 
                             base_agent, merchant, location, context_clues[base_agent])

        # Check for travel notice related context
        for country in self.travel_notice_data.get("countries", []):
            if re.search(r'\b' + re.escape(country.lower()) + r'\b', prompt_lower):
                context_clues["TravelNoticeAgent"] = context_clues.get("TravelNoticeAgent", 0) + 2
                logger.debug("Travel context matched: country=%s, score=%d", 
                             country, context_clues.get("TravelNoticeAgent", 0))

        if re.search(r"\btravel\b", prompt_lower) and re.search(r"\bnotice\b", prompt_lower):
            context_clues["TravelNoticeAgent"] = context_clues.get("TravelNoticeAgent", 0) + 3
            logger.debug("Travel notice keywords matched, score=%d", 
                         context_clues.get("TravelNoticeAgent", 0))

        # Check for card-related context
        if re.search(r"\blost\b", prompt_lower) and re.search(r"\bcard\b", prompt_lower):
            context_clues["CardServicesAgent"] = context_clues.get("CardServicesAgent", 0) + 3
            logger.debug("Card loss context matched, score=%d", 
                         context_clues.get("CardServicesAgent", 0))

        # Check for specific location keywords hinting at travel
        travel_locations = ["tokyo", "japan", "berlin", "germany", "barcelona", "spain"]
        for loc in travel_locations:
            if re.search(r'\b' + loc + r'\b', prompt_lower):
                context_clues["TravelNoticeAgent"] = context_clues.get("TravelNoticeAgent", 0) + 1
                logger.debug("Travel location matched: %s, score=%d", 
                             loc, context_clues.get("TravelNoticeAgent", 0))

        return context_clues

    def _make_groq_request(self, messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """Makes a request to the Groq API."""
        logger.debug("Making Groq API request with %d messages", len(messages))
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.error("Groq API error: %s - %s", response.status_code, response.text)
                return None, f"API error: {response.status_code} - {response.text}"
            content = response.json()["choices"][0]["message"]["content"].strip()
            logger.info("Groq API request successful")
            return content, None
        except requests.RequestException as e:
            logger.error("Network error in Groq API request: %s", str(e))
            return None, f"Network error: {str(e)}"
        except Exception as e:
            logger.error("Unexpected error in Groq API request: %s", str(e))
            return None, f"Unexpected error: {str(e)}"