from typing import Dict, List, Any, Optional, Tuple
import datetime
import re
import json
import logging
import openai

# Use the same logger as sai.py and agent_router.py
logger = logging.getLogger('ChainOfThought')

class BaseAgent:
    """
    Base class for all specialized agents.
    Includes methods to log reasoning steps during processing.
    Supports optional OpenAI integration for enhanced processing.
    """

    def __init__(self, customer_data: Dict, travel_notice_data: Dict, recent_transactions: List[Dict], openai_api_key: Optional[str] = None, model: Optional[str] = None):
        self.customer_data = customer_data
        self.travel_notice_data = travel_notice_data
        self.recent_transactions = recent_transactions
        self.openai_api_key = openai_api_key
        self.model = model
        self.reasoning_log = {
            "agent_type": self.__class__.__name__,
            "analysis_steps": [],
            "decision_factors": {},
            "actions_considered": [],
            "actions_taken": [],
            "response_construction": "",
            "next_best_actions": []
        }
        logger.info("%s initialized with OpenAI: %s", self.__class__.__name__, "enabled" if self.openai_api_key else "disabled")

    def process(self, user_prompt: str) -> Dict:
        """
        Process the user prompt and return a response including the reasoning log.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process method")

    def _add_analysis_step(self, step_description: str):
        """Add a step to the reasoning log"""
        self.reasoning_log["analysis_steps"].append(step_description)
        logger.debug("%s - Analysis step: %s", self.__class__.__name__, step_description)

    def _add_decision_factor(self, factor: str, value: Any):
        """Add a decision factor to the reasoning log"""
        if isinstance(value, (datetime.date, datetime.datetime)):
            value = value.isoformat()
        self.reasoning_log["decision_factors"][factor] = value
        logger.debug("%s - Decision factor: %s = %s", self.__class__.__name__, factor, value)

    def _consider_action(self, action: str, reason: str):
        """Add an action under consideration to the reasoning log"""
        self.reasoning_log["actions_considered"].append({"action": action, "reason": reason})
        logger.debug("%s - Considered action: %s (Reason: %s)", self.__class__.__name__, action, reason)

    def _take_action(self, action: str, details: str):
        """Add an action taken to the reasoning log"""
        self.reasoning_log["actions_taken"].append({"action": action, "details": details})
        logger.info("%s - Action taken: %s - %s", self.__class__.__name__, action, details)

    def _set_response_construction(self, explanation: str):
        """Set the response construction reasoning"""
        self.reasoning_log["response_construction"] = explanation
        logger.debug("%s - Response construction: %s", self.__class__.__name__, explanation)

    def _add_next_best_action(self, action: str, priority: str, description: str, category: str, icon: str = '🔹'):
        """Add a next best action recommendation"""
        if "next_best_actions" not in self.reasoning_log:
            self.reasoning_log["next_best_actions"] = []
        self.reasoning_log["next_best_actions"].append({
            "action": action,
            "priority": priority,
            "description": description,
            "category": category,
            "icon": icon
        })
        logger.info("%s - Added next best action: %s (Priority: %s, Category: %s)", 
                    self.__class__.__name__, action, priority, category)

    def _make_openai_request(self, messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """Makes a request to the OpenAI API using the updated SDK syntax."""
        if not self.openai_api_key or not self.model:
            return None, "OpenAI not configured: No API key or model provided."
        logger.debug("%s - Making OpenAI API request with %d messages", self.__class__.__name__, len(messages))
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            content = response.choices[0].message.content.strip()
            logger.info("%s - OpenAI API request successful", self.__class__.__name__)
            return content, None
        except openai.AuthenticationError:
            logger.error("%s - Invalid OpenAI API key", self.__class__.__name__)
            return None, "Invalid API key. Please check your OpenAI API key."
        except openai.RateLimitError:
            logger.error("%s - OpenAI API rate limit exceeded", self.__class__.__name__)
            return None, "Rate limit exceeded. Please try again later."
        except openai.OpenAIError as e:
            logger.error("%s - OpenAI API error: %s", self.__class__.__name__, str(e))
            return None, f"API error: {str(e)}"
        except Exception as e:
            logger.error("%s - Unexpected error in OpenAI API request: %s", self.__class__.__name__, str(e))
            return None, f"Unexpected error: {str(e)}"


class TransactionAnalysisAgent(BaseAgent):
    """Agent specializing in transaction analysis and resolution"""

    def process(self, user_prompt: str) -> Dict:
        logger.info("%s processing prompt: %s", self.__class__.__name__, user_prompt)
        self._add_analysis_step("Initializing TransactionAnalysisAgent")
        self._add_analysis_step(f"Received user prompt: '{user_prompt}'")

        if self.openai_api_key and self.model:
            return self._process_with_openai(user_prompt)
        else:
            return self._process_rule_based(user_prompt)

    def _process_with_openai(self, user_prompt: str) -> Dict:
        """Process the prompt using OpenAI for intent detection and response generation."""
        self._add_analysis_step("Using OpenAI for transaction analysis")
        prompt = """
You are a banking transaction analysis agent. Your task is to analyze a customer query about their transactions and provide a response with recommended actions.

**Context Data**:
- Customer Info: {customer_data}
- Recent Transactions: {recent_transactions}
- Travel Notices: {travel_notice_data}
- Customer Query: {query}

**Instructions**:
1. Analyze the query to identify the intent (e.g., check declined transaction, review recent purchases).
2. Examine the recent transactions and travel notices for relevant details (e.g., declines, locations).
3. Generate a natural language response addressing the query.
4. Suggest 1-3 next best actions with priority (High, Medium, Low), description, category, and an emoji icon.
5. Return a JSON object with:
   - "response": The natural language response.
   - "next_best_actions": List of actions, each with "action", "priority", "description", "category", "icon".
   - "intent": The detected intent.
Return ONLY the JSON object, wrapped in ```json\n...\n```.
"""
        messages = [{
            "role": "system",
            "content": prompt.format(
                customer_data=json.dumps(self.customer_data, indent=2),
                recent_transactions=json.dumps(self.recent_transactions, indent=2),
                travel_notice_data=json.dumps(self.travel_notice_data, indent=2),
                query=user_prompt
            )
        }]
        response, error = self._make_openai_request(messages)
        if error:
            logger.error("%s - OpenAI error: %s. Falling back to rule-based processing", self.__class__.__name__, error)
            self._add_analysis_step(f"OpenAI error: {error}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

        try:
            # Strip JSON code fences
            content = response.strip()
            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content[7:-4].strip()
            result = json.loads(content)
            self._add_decision_factor("openai_detected_intent", result.get("intent", "unknown"))
            self._add_analysis_step(f"OpenAI detected intent: {result.get('intent', 'unknown')}")
            self._set_response_construction("Response generated by OpenAI based on query analysis")
            for action in result.get("next_best_actions", []):
                self._add_next_best_action(
                    action["action"],
                    action["priority"],
                    action["description"],
                    action["category"],
                    action.get("icon", "🔹")
                )
            self._take_action("OpenAI Response Generated", f"Processed query: {user_prompt}")
            return {
                "response": result["response"],
                "reasoning_log": self.reasoning_log,
                "next_best_actions": result.get("next_best_actions", [])
            }
        except json.JSONDecodeError as e:
            logger.error("%s - Invalid JSON from OpenAI: %s. Falling back to rule-based", self.__class__.__name__, str(e))
            self._add_analysis_step(f"Invalid JSON from OpenAI: {str(e)}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

    def _process_rule_based(self, user_prompt: str) -> Dict:
        """Original rule-based processing logic."""
        self._add_analysis_step("Using rule-based transaction analysis")

        # Extract relevant transaction mentions
        self._add_analysis_step("Identifying specific transaction details (locations, merchants) mentioned in the query")
        mentioned_locations = self._extract_locations(user_prompt)
        mentioned_merchants = self._extract_merchants(user_prompt)
        mentioned_explicitly = bool(mentioned_locations or mentioned_merchants)

        self._add_decision_factor("mentioned_locations", mentioned_locations)
        self._add_decision_factor("mentioned_merchants", mentioned_merchants)
        self._add_decision_factor("user_mentioned_specific_transaction", mentioned_explicitly)

        # Find relevant transactions
        self._add_analysis_step("Searching for relevant transactions in recent customer history")
        relevant_transactions = []

        if mentioned_explicitly:
            for tx in self.recent_transactions:
                location_match = any(loc.lower() in tx.get("location", "").lower() for loc in mentioned_locations)
                merchant_match = any(merch.lower() in tx.get("merchant", "").lower() for merch in mentioned_merchants)
                if location_match or merchant_match:
                    relevant_transactions.append(tx)
            self._add_analysis_step(f"Found {len(relevant_transactions)} transactions matching explicit mentions.")

        is_decline_focused = re.search(r"\b(decline[d]?|denied|rejected|failed)\b", user_prompt, re.IGNORECASE)
        is_general_transaction_query = re.search(r"\b(transaction[s]?|purchase[s]?|charge[s]?|payment[s]?)\b", user_prompt, re.IGNORECASE) and not mentioned_explicitly

        if not relevant_transactions and is_decline_focused:
            self._add_analysis_step("No specific transaction match found, but user mentioned 'declined'. Searching for all recent declined transactions.")
            relevant_transactions = [tx for tx in self.recent_transactions if tx.get("status", "").lower() == "declined"]
            self._add_decision_factor("search_mode", "all_declined")
            self._add_analysis_step(f"Found {len(relevant_transactions)} declined transactions.")

        if not relevant_transactions and is_general_transaction_query and not is_decline_focused:
            self._add_analysis_step("No specific or declined transactions found, but query mentions transactions generally. Fetching last 3 transactions.")
            relevant_transactions = self.recent_transactions[:3]
            self._add_decision_factor("search_mode", "last_3_general")
            self._add_analysis_step(f"Showing the {len(relevant_transactions)} most recent transactions.")

        self._add_decision_factor("relevant_transactions_identified", relevant_transactions)

        declined_transactions = [tx for tx in relevant_transactions if tx.get("status", "").lower() == "declined"]
        self._add_decision_factor("declined_transactions_analyzed", declined_transactions)

        response = ""

        if declined_transactions:
            self._add_analysis_step("Analyzing reasons for identified declined transactions")
            for tx in declined_transactions:
                self._consider_action(
                    f"Explain declined transaction: {tx.get('amount')} at {tx.get('merchant')}",
                    f"Transaction declined on {tx.get('date')} due to: {tx.get('reason')}"
                )

            self._set_response_construction("Building response focused on explaining declined transactions and offering solutions")

            if len(declined_transactions) == 1:
                tx = declined_transactions[0]
                self._take_action(
                    "Transaction Explanation Provided",
                    f"Explained single declined transaction: {tx.get('amount')} at {tx.get('merchant')} (Reason: {tx.get('reason')})"
                )
                response = f"I looked into the transaction you mentioned. The purchase of {tx.get('amount')} at {tx.get('merchant')} in {tx.get('location')} on {tx.get('date')} was declined because '{tx.get('reason')}'."

                reason_lower = tx.get('reason', '').lower()
                if "insufficient funds" in reason_lower:
                    response += " This often happens if the account balance wasn't enough at the moment of the purchase."
                    self._add_next_best_action(
                        "Set Balance Alerts", "Medium",
                        "Suggest setting up low balance alerts to help avoid this in the future.",
                        "Account Management", "💰"
                    )
                    self._add_next_best_action(
                        "Check Available Balance", "Medium",
                        "Offer to check the current available balance.",
                        "Account Management", "📊"
                    )

                elif "card reported lost" in reason_lower:
                    response += " This was because the card used was marked as lost or stolen in our system. If you have found this card, we need to reactivate it or issue a new one."
                    self._add_next_best_action(
                        "Review Card Status", "High",
                        "Verify if the customer's card needs reactivation or replacement.",
                        "Card Services", "💳"
                    )
                    self._add_next_best_action(
                        "Issue Replacement Card", "High",
                        "Offer to immediately issue a replacement card.",
                        "Card Services", "🆕"
                    )

                elif "incorrect pin" in reason_lower or "pin attempts exceeded" in reason_lower:
                    response += " The decline was due to an incorrect PIN entry. If you've forgotten your PIN, I can help you reset it."
                    self._add_next_best_action(
                        "Reset PIN", "High",
                        "Offer to guide the customer through the PIN reset process.",
                        "Card Services", "🔑"
                    )

                elif "travel notice" in reason_lower or "unusual activity" in reason_lower or "security block" in reason_lower:
                    location = tx.get('location', '').lower()
                    is_international = any(loc in location for loc in ["japan", "germany", "spain", "france", "italy"])

                    if is_international:
                        response += f" This decline appears to be related to international usage in {tx.get('location', 'that location')}. Often, setting a travel notice helps prevent this."
                        notice_countries = [c.lower() for c in self.travel_notice_data.get("countries", [])]
                        notice_covers_location = any(loc in location for loc in notice_countries)

                        if not self.travel_notice_data or not notice_covers_location:
                            response += " I don't see an active travel notice covering this location."
                            self._add_next_best_action(
                                "Create/Update Travel Notice", "High",
                                f"Offer to set up or update a travel notice to include {tx.get('location', 'this region')}.",
                                "Travel Services", "✈️"
                            )
                        else:
                            response += " Although you have a travel notice, the transaction was still flagged. Let's review the details to ensure everything is set correctly."
                            self._add_next_best_action(
                                "Review Travel Notice Details", "Medium",
                                "Verify the dates and countries on the existing travel notice.",
                                "Travel Services", "🔍"
                            )

                    else:
                        response += " The transaction was flagged due to unusual activity patterns for security reasons. Verifying the transaction can help prevent future issues."
                        self._add_next_best_action(
                            "Verify Transaction", "High",
                            "Ask the customer to confirm if the flagged transaction was legitimate.",
                            "Security", "✅"
                        )
                        self._add_next_best_action(
                            "Review Recent Activity", "Medium",
                            "Offer to review other recent transactions for any unrecognized activity.",
                            "Security", "🛡️"
                        )

                else:
                    response += " There might be a few reasons why this could happen."
                    self._add_next_best_action(
                        "Detailed Transaction Review", "Medium",
                        "Offer to investigate the specific reason for this decline further.",
                        "Account Management", "ℹ️"
                    )

            else:
                self._take_action(
                    "Multiple Declined Transactions Summarized",
                    f"Summarized {len(declined_transactions)} declined transactions."
                )
                response = f"I found {len(declined_transactions)} recently declined transactions matching your query:\n"
                for tx in declined_transactions[:3]:
                    response += f"- {tx.get('date')}: {tx.get('amount')} at {tx.get('merchant')} in {tx.get('location')} (Reason: {tx.get('reason')})\n"
                if len(declined_transactions) > 3:
                    response += "- ... and possibly others.\n"

                response += "\nWould you like me to go over these declines in more detail, or perhaps look into a specific one?"
                self._add_next_best_action(
                    "Review All Declined Transactions", "Medium",
                    "Offer to review all recent declined transactions and their reasons.",
                    "Account Management", "📋"
                )
                self._add_next_best_action(
                    "Check Specific Declined Transaction", "Medium",
                    "Ask the customer if they want details about a particular declined transaction.",
                    "Account Management", "❓"
                )

        elif relevant_transactions:
            self._add_analysis_step("No relevant declined transactions found, but other matching transactions were identified.")
            self._set_response_construction("Building response about the specific approved transactions found.")
            self._take_action(
                "Specific Transaction(s) Information Provided",
                f"Provided details for {len(relevant_transactions)} approved transactions matching user query."
            )

            response = "I found these transactions matching your description:\n"
            for tx in relevant_transactions[:5]:
                response += f"- {tx.get('date')}: {tx.get('amount')} at {tx.get('merchant')} in {tx.get('location')} (Status: {tx.get('status')})\n"

            if all(tx.get('status', '').lower() == 'approved' for tx in relevant_transactions):
                response += "\nIt looks like these were all approved successfully. Did you have a specific question about them?"
                self._add_next_best_action(
                    "Clarify Transaction Question", "Low",
                    "Ask the user for more details about their question regarding these transactions.",
                    "General Inquiry", "🤔"
                )
            else:
                response += "\nLet me know if you need more details on any of these."

        else:
            self._add_analysis_step("No transactions found matching the query specifics or general transaction keywords.")
            self._set_response_construction("No relevant transactions found, providing general response and options.")
            self._take_action(
                "No Matching Transactions Found",
                "Informed user that no matching transactions were found."
            )
            response = "I couldn't find  find any recent transactions specifically matching your description."

            if is_decline_focused:
                response += " Were you looking for a declined transaction from a while ago, or perhaps at a different place?"
                self._add_next_best_action(
                    "Search Older Transactions", "Low",
                    "Offer to search transaction history further back (e.g., 60 or 90 days).",
                    "Account Management", "📅"
                )
                self._add_next_best_action(
                    "Verify Merchant/Location", "Low",
                    "Ask the user to confirm the spelling or details of the merchant/location.",
                    "General Inquiry", "✍️"
                )
            else:
                response += " Would you like to see your full recent transaction history, or search by a specific date range?"
                self._add_next_best_action(
                    "Show Full Recent History", "Low",
                    "Offer to display the complete transaction history for the last 30 days.",
                    "Account Management", "📜"
                )
                self._add_next_best_action(
                    "Search by Date Range", "Low",
                    "Offer to search for transactions within a specific start and end date.",
                    "Account Management", "🗓️"
                )

        logger.info("%s completed rule-based processing", self.__class__.__name__)
        return {
            "response": response.strip(),
            "reasoning_log": self.reasoning_log,
            "next_best_actions": self.reasoning_log.get("next_best_actions", [])
        }

    def _extract_locations(self, user_prompt: str) -> List[str]:
        """Extract mentioned locations from the user prompt using regex for robustness"""
        logger.debug("%s extracting locations from prompt", self.__class__.__name__)
        locations = []
        prompt_lower = user_prompt.lower()

        location_keywords = [
            "japan", "tokyo", "osaka",
            "germany", "berlin", "munich",
            "spain", "barcelona", "madrid",
            "france", "paris", "nice",
            "italy", "rome", "milan",
            "usa", "us", "united states",
            "uk", "united kingdom", "london",
        ]

        for location in location_keywords:
            if re.search(r'\b' + re.escape(location) + r'\b', prompt_lower):
                if location in ["usa", "us", "united states"]:
                    normalized_location = "USA"
                elif location in ["uk", "united kingdom"]:
                    normalized_location = "UK"
                else:
                    normalized_location = location.title()

                if normalized_location not in locations:
                    locations.append(normalized_location)

        return locations

    def _extract_merchants(self, user_prompt: str) -> List[str]:
        """Extract mentioned merchants from the user prompt using regex"""
        logger.debug("%s extracting merchants from prompt", self.__class__.__name__)
        merchants = []
        prompt_lower = user_prompt.lower()

        merchant_keywords = [
            "starbucks", "tokyo central", "market", "uber", "lyft", "la casa",
            "tapas", "amazon", "marketplace", "target", "walmart", "cvs",
            "walgreens", "best buy", "home depot", "costco", "safeway",
            "whole foods",
            "delta", "united airlines", "marriott", "hilton", "ebay"
        ]

        for merchant in merchant_keywords:
            escaped_merchant = re.escape(merchant)
            if re.search(r'\b' + escaped_merchant + r'\b', prompt_lower):
                if merchant not in merchants:
                    merchants.append(merchant)

        return merchants


class TravelNoticeAgent(BaseAgent):
    """Agent specializing in travel notices and international transactions"""

    def process(self, user_prompt: str) -> Dict:
        logger.info("%s processing prompt: %s", self.__class__.__name__, user_prompt)
        self._add_analysis_step("Initializing TravelNoticeAgent")
        self._add_analysis_step(f"Received user prompt: '{user_prompt}'")

        if self.openai_api_key and self.model:
            return self._process_with_openai(user_prompt)
        else:
            return self._process_rule_based(user_prompt)

    def _process_with_openai(self, user_prompt: str) -> Dict:
        """Process the prompt using OpenAI for intent detection and response generation."""
        self._add_analysis_step("Using OpenAI for travel notice processing")
        prompt = """
You are a banking travel notice agent. Your task is to analyze a customer query about travel notices or international transactions and provide a response with recommended actions.

**Context Data**:
- Customer Info: {customer_data}
- Recent Transactions: {recent_transactions}
- Travel Notices: {travel_notice_data}
- Customer Query: {query}

**Instructions**:
1. Analyze the query to identify the intent (e.g., check travel notice status, create new notice, update notice, activate notice).
2. Check the travel notice data for active notices and relevant details.
3. Generate a natural language response addressing the query.
4. Suggest 1-3 next best actions with priority (High, Medium, Low), description, category, and an emoji icon.
5. Return a JSON object with:
   - "response": The natural language response.
   - "next_best_actions": List of actions, each with "action", "priority", "description", "category", "icon".
   - "intent": The detected intent.
Return ONLY the JSON object, wrapped in ```json\n...\n```.
"""
        messages = [{
            "role": "system",
            "content": prompt.format(
                customer_data=json.dumps(self.customer_data, indent=2),
                recent_transactions=json.dumps(self.recent_transactions, indent=2),
                travel_notice_data=json.dumps(self.travel_notice_data, indent=2),
                query=user_prompt
            )
        }]
        response, error = self._make_openai_request(messages)
        if error:
            logger.error("%s - OpenAI error: %s. Falling back to rule-based processing", self.__class__.__name__, error)
            self._add_analysis_step(f"OpenAI error: {error}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

        try:
            # Strip JSON code fences
            content = response.strip()
            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content[7:-4].strip()
            result = json.loads(content)
            self._add_decision_factor("openai_detected_intent", result.get("intent", "unknown"))
            self._add_analysis_step(f"OpenAI detected intent: {result.get('intent', 'unknown')}")
            self._set_response_construction("Response generated by OpenAI based on query analysis")
            for action in result.get("next_best_actions", []):
                self._add_next_best_action(
                    action["action"],
                    action["priority"],
                    action["description"],
                    action["category"],
                    action.get("icon", "🔹")
                )
            self._take_action("OpenAI Response Generated", f"Processed query: {user_prompt}")
            return {
                "response": result["response"],
                "reasoning_log": self.reasoning_log,
                "next_best_actions": result.get("next_best_actions", [])
            }
        except json.JSONDecodeError as e:
            logger.error("%s - Invalid JSON from OpenAI: %s. Falling back to rule-based", self.__class__.__name__, str(e))
            self._add_analysis_step(f"Invalid JSON from OpenAI: {str(e)}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

    def _process_rule_based(self, user_prompt: str) -> Dict:
        """Original rule-based processing logic."""
        self._add_analysis_step("Using rule-based travel notice processing")

        intent = self._determine_intent(user_prompt)
        self._add_decision_factor("determined_intent", intent)
        self._add_analysis_step(f"Determined user intent: {intent}")

        mentioned_countries = self._extract_countries(user_prompt)
        self._add_decision_factor("mentioned_countries", mentioned_countries)
        self._add_analysis_step(f"Extracted countries from prompt: {mentioned_countries}")

        active_notice, notice_details = self._check_active_notice()
        self._add_decision_factor("has_active_notice", active_notice)
        self._add_decision_factor("current_notice_details", notice_details)
        self._add_analysis_step(f"Checked for active travel notice. Active: {active_notice}, Details: {notice_details}")

        response = ""
        needs_activation_fix = active_notice and notice_details.get('status') == "Submitted but not activated due to system error"

        if intent == "check_status":
            self._add_analysis_step("Handling 'check_status' intent.")
            self._consider_action("Provide current travel notice status", "User asked about existing travel notice.")

            if active_notice:
                self._take_action("Travel Notice Status Report", f"Provided details about active notice for {notice_details.get('countries')}.")
                self._set_response_construction("Informing user about active travel notice details.")
                response = f"You currently have an active travel notice set for {', '.join(notice_details.get('countries', []))} from {notice_details.get('travel_start', 'N/A')} to {notice_details.get('travel_end', 'N/A')}."

                if needs_activation_fix:
                    response += " However, I see it wasn't activated correctly due to a system issue."
                    self._add_next_best_action(
                        "Fix & Activate Notice", "High",
                        "Immediately fix the system error and activate the pending travel notice.",
                        "Travel Services", "🛠️"
                    )
                    response += " I can fix that for you right now."
                else:
                    response += " Your card should work as expected in these locations during this period."
                    self._add_next_best_action(
                        "View Notice Details", "Low",
                        "Offer to show the full details of the active travel notice.",
                        "Travel Services", "📄"
                    )

            else:
                self._take_action("Travel Notice Status Report", "Informed user no active notice found.")
                self._set_response_construction("Informing user they have no active travel notice.")
                response = "It looks like you don't have any active travel notices set up right now."
                self._add_next_best_action(
                    "Create Travel Notice", "Medium",
                    "Offer to help create a new travel notice for an upcoming trip.",
                    "Travel Services", "➕"
                )
                response += " Are you planning a trip? I can help you set one up."

        elif intent == "create_notice":
            self._add_analysis_step("Handling 'create_notice' intent.")
            self._consider_action("Set up new travel notice", "User expressed intent to create a travel plan/notice.")

            if active_notice:
                self._take_action("Create Notice Halted", "User already has an active notice.")
                self._set_response_construction("Informing about existing notice and offering to update it instead.")
                response = f"You already have a travel notice active for {', '.join(notice_details.get('countries', []))} (until {notice_details.get('travel_end', 'N/A')}). Did you want to update this existing notice, perhaps add more countries or change the dates?"
                self._add_next_best_action(
                    "Update Existing Notice", "Medium",
                    "Offer to modify the current travel notice instead of creating a new one.",
                    "Travel Services", "✏️"
                )
                self._add_next_best_action(
                    "Cancel Existing Notice", "Low",
                    "Offer to cancel the current notice if it's no longer needed.",
                    "Travel Services", "❌"
                )
            else:
                self._take_action("New Travel Notice Creation Initiated", "Guiding user through the creation process.")
                self._set_response_construction("Guiding user through creating a new travel notice, prompting for necessary details.")
                response = "Okay, I can help you set up a new travel notice. To do this, I'll need a few details:"
                response += "\n- Which countries will you be visiting?"
                response += "\n- What is the start date of your trip?"
                response += "\n- What is the end date of your trip?"

                if mentioned_countries:
                    response += f"\n\nYou mentioned {', '.join(mentioned_countries)}. Shall I include these in the notice?"
                    self._add_next_best_action(
                        "Confirm Countries for Notice", "High",
                        f"Ask user to confirm adding {', '.join(mentioned_countries)} to the new notice.",
                        "Travel Services", "✅"
                    )
                else:
                    self._add_next_best_action(
                        "Provide Travel Details", "High",
                        "Prompt user to provide countries, start date, and end date.",
                        "Travel Services", "❓"
                    )

        elif intent == "update_notice":
            self._add_analysis_step("Handling 'update_notice' intent.")
            self._consider_action("Update existing travel notice", "User expressed intent to modify their travel notice.")

            if active_notice:
                self._take_action("Travel Notice Update Initiated", f"Offering to update notice for {notice_details.get('countries')}.")
                self._set_response_construction("Guiding user through updating their existing notice, asking what needs changing.")
                response = f"Sure, I can help update your current travel notice (for {', '.join(notice_details.get('countries', []))}, valid until {notice_details.get('travel_end', 'N/A')}). What would you like to change? You can add/remove countries or adjust the travel dates."

                not_included = [country for country in mentioned_countries if country not in notice_details.get('countries', [])]
                already_included = [country for country in mentioned_countries if country in notice_details.get('countries', [])]

                if not_included:
                    response += f"\n\nI see you mentioned {', '.join(not_included)}. Would you like to add them to the notice?"
                    self._add_next_best_action(
                        "Add Countries to Notice", "Medium",
                        f"Offer to add {', '.join(not_included)} to the existing travel notice.",
                        "Travel Services", "➕"
                    )
                elif already_included:
                    response += f"\n\nYou mentioned {', '.join(already_included)}, which are already included. Did you want to change the travel dates associated with this notice?"
                    self._add_next_best_action(
                        "Update Travel Dates", "Medium",
                        "Ask user if they want to modify the start or end dates for the notice.",
                        "Travel Services", "📅"
                    )
                else:
                    self._add_next_best_action(
                        "Specify Notice Changes", "High",
                        "Ask the user to specify what they want to change (countries or dates).",
                        "Travel Services", "❓"
                    )

            else:
                self._take_action("Update Notice Halted", "No active notice found to update.")
                self._set_response_construction("No active notice to update, offering to create a new one instead.")
                response = "It looks like you don't have an active travel notice to update right now. Would you like to create a new one for an upcoming trip instead?"
                self._add_next_best_action(
                    "Create Travel Notice", "Medium",
                    "Guide customer through creating a new travel notice since none exists to update.",
                    "Travel Services", "➕"
                )

        elif intent == "activate_notice":
            self._add_analysis_step("Handling 'activate_notice' intent.")
            self._consider_action("Activate pending/faulty travel notice", "User specifically asked to activate or fix their notice.")

            if needs_activation_fix:
                self._take_action("Travel Notice Activation (Fix)", "Fixed system error and activated the travel notice.")
                self._set_response_construction("Confirming notice activation after fixing the system error and apologizing.")
                self.travel_notice_data['status'] = 'Active'
                notice_details['status'] = 'Active'

                response = "I found the issue! There was a system glitch preventing your notice from activating properly. I've fixed that now."
                response += f"\nYour travel notice for {', '.join(notice_details.get('countries', []))} (from {notice_details.get('travel_start', 'N/A')} to {notice_details.get('travel_end', 'N/A')}) is now **active**. "
                response += "Apologies for that error. Your card should now work correctly in those locations."

                self._add_next_best_action(
                    "Confirm Recent Declines Resolved", "High",
                    "Ask if the customer experienced any declines recently that should now be resolved.",
                    "Card Services", "👍"
                )
            elif active_notice:
                self._take_action("Activation Check Complete", "Notice already active.")
                self._set_response_construction("Informing user that their travel notice is already active.")
                response = f"Good news! Your travel notice for {', '.join(notice_details.get('countries', []))} is already active and runs until {notice_details.get('travel_end', 'N/A')}. No further action needed on activation."
            else:
                self._take_action("Activation Halted", "No notice found to activate.")
                self._set_response_construction("No notice found to activate, offering to create one.")
                response = "I couldn't find a pending travel notice to activate. Do you need help setting up a new travel notice for a trip?"
                self._add_next_best_action(
                    "Create Travel Notice", "Medium",
                    "Guide customer through creating a new travel notice as none exists to activate.",
                    "Travel Services", "➕"
                )

        self._add_analysis_step("Performing post-intent analysis: Checking for transaction/notice mismatches.")
        mismatch_found = False
        if active_notice and notice_details.get('status', '').lower() == 'active':
            countries_in_notice = [c.lower() for c in notice_details.get('countries', [])]
            countries_with_declines_outside_notice = set()

            for tx in self.recent_transactions:
                if tx.get('status', '').lower() == 'declined':
                    location = tx.get('location', '').lower()
                    tx_country = None

                    known_countries = ['japan', 'germany', 'spain', 'usa', 'france', 'italy', 'uk']
                    for country in known_countries:
                        if country in location:
                            tx_country = country
                            break

                    if tx_country and tx_country not in countries_in_notice:
                        reason = tx.get('reason', '').lower()
                        if "travel" in reason or "location" in reason or "security block" in reason or "unusual activity" in reason:
                            countries_with_declines_outside_notice.add(tx_country.title())
                            mismatch_found = True

            if countries_with_declines_outside_notice:
                self._add_analysis_step(f"Found declined transactions in countries not covered by the active notice: {countries_with_declines_outside_notice}")
                self._add_next_best_action(
                    "Add Country to Notice", "High",
                    f"Suggest adding {', '.join(countries_with_declines_outside_notice)} to the travel notice due to recent declines.",
                    "Travel Services", "🌍"
                )
                if response and not response.endswith("?"):
                    response += "\n\n**Additionally:** I noticed you had recent declined transactions in "
                    response += f"{', '.join(countries_with_declines_outside_notice)}, which aren't currently covered by your active travel notice. "
                    response += "Adding these countries could prevent future declines there. Would you like to do that?"
                elif not response:
                    response = "I noticed you had recent declined transactions in "
                    response += f"{', '.join(countries_with_declines_outside_notice)}, which aren't currently covered by your active travel notice. "
                    response += "Adding these countries could prevent future declines there. Would you like to do that?"

        if not mismatch_found:
            self._add_analysis_step("No transaction/notice mismatches found requiring immediate action.")

        logger.info("%s completed rule-based processing", self.__class__.__name__)
        return {
            "response": response.strip(),
            "reasoning_log": self.reasoning_log,
            "next_best_actions": self.reasoning_log.get("next_best_actions", [])
        }

    def _determine_intent(self, user_prompt: str) -> str:
        """Determine the primary user intent regarding travel notices"""
        logger.debug("%s determining intent from prompt", self.__class__.__name__)
        prompt_lower = user_prompt.lower()

        if re.search(r"\b(activate|fix|enable|reactivate|confirm it[']?s active)\b", prompt_lower):
            if "activate travel" in prompt_lower and not any(k in prompt_lower for k in ["fix", "enable", "confirm"]):
                pass
            else:
                self._add_analysis_step("Detected keywords related to activating or fixing a notice.")
                return "activate_notice"

        create_keywords = ["create", "set up", "new", "add.*notice", "submit.*notice", "inform.*travel", "going to"]
        if any(re.search(r'\b' + keyword + r'\b', prompt_lower) for keyword in create_keywords):
            if not re.search(r"\b(update|change|modify|edit)\b", prompt_lower):
                self._add_analysis_step("Detected keywords related to creating a new notice.")
                return "create_notice"

        if re.search(r"\b(update|change|modify|edit|add countries|remove countries|extend|shorten)\b", prompt_lower):
            self._add_analysis_step("Detected keywords related to updating an existing notice.")
            return "update_notice"

        self._add_analysis_step("No specific create/update/activate keywords found. Defaulting intent to 'check_status'.")
        return "check_status"

    def _extract_countries(self, user_prompt: str) -> List[str]:
        """Extract mentioned countries from the user prompt"""
        logger.debug("%s extracting countries from prompt", self.__class__.__name__)
        countries = []
        prompt_lower = user_prompt.lower()
        country_keywords = [
            "japan", "tokyo", "osaka",
            "germany", "berlin", "munich",
            "spain", "barcelona", "madrid",
            "france", "paris", "nice",
            "italy", "rome", "milan",
            "usa", "us", "united states", "america",
            "uk", "united kingdom", "london", "england", "scotland",
            "canada", "mexico", "australia",
        ]
        for country in country_keywords:
            if re.search(r'\b' + re.escape(country) + r'\b', prompt_lower):
                if country in ["usa", "us", "united states", "america"]:
                    normalized = "USA"
                elif country in ["uk", "united kingdom", "england", "scotland"]:
                    normalized = "UK"
                else:
                    normalized = country.title()

                if normalized not in countries:
                    countries.append(normalized)
        return countries

    def _check_active_notice(self) -> Tuple[bool, Dict]:
        """Check if there's an active travel notice and return its details"""
        logger.debug("%s checking active travel notice", self.__class__.__name__)
        if not self.travel_notice_data or not self.travel_notice_data.get('countries'):
            return False, {}

        details = self.travel_notice_data.copy()

        try:
            start_date_str = details.get('travel_start', '')
            end_date_str = details.get('travel_end', '')
            if not start_date_str or not end_date_str:
                return bool(details.get('countries')), details

            date_format = "%B %d, %Y"
            start_date = datetime.datetime.strptime(start_date_str, date_format).date()
            end_date = datetime.datetime.strptime(end_date_str, date_format).date()
            today = datetime.date.today()

            is_active_time = start_date <= today <= end_date or start_date > today
            status = details.get('status', 'Unknown').lower()
            is_active_status = "active" in status or "submitted" in status

            is_active = is_active_time and is_active_status
            return is_active, details

        except ValueError:
            self._add_analysis_step("Warning: Failed to parse travel notice dates. Activity status based solely on presence of countries.")
            return bool(details.get('countries')), details


class CardServicesAgent(BaseAgent):
    """Agent specializing in card-related services"""

    def process(self, user_prompt: str) -> Dict:
        logger.info("%s processing prompt: %s", self.__class__.__name__, user_prompt)
        self._add_analysis_step("Initializing CardServicesAgent")
        self._add_analysis_step(f"Received user prompt: '{user_prompt}'")

        if self.openai_api_key and self.model:
            return self._process_with_openai(user_prompt)
        else:
            return self._process_rule_based(user_prompt)

    def _process_with_openai(self, user_prompt: str) -> Dict:
        """Process the prompt using OpenAI for intent detection and response generation."""
        self._add_analysis_step("Using OpenAI for card services processing")
        prompt = """
You are a banking card services agent. Your task is to analyze a customer query about their card (e.g., status, lost/stolen, replacement, limits) and provide a response with recommended actions.

**Context Data**:
- Customer Info: {customer_data}
- Recent Transactions: {recent_transactions}
- Travel Notices: {travel_notice_data}
- Customer Query: {query}

**Instructions**:
1. Analyze the query to identify the intent (e.g., check card status, report lost/stolen, replace card, check limits).
2. Check the customer data and transactions for relevant card details (e.g., status, declines).
3. Generate a natural language response addressing the query.
4. Suggest 1-3 next best actions with priority (High, Medium, Low), description, category, and an emoji icon.
5. Return a JSON object with:
   - "response": The natural language response.
   - "next_best_actions": List of actions, each with "action", "priority", "description", "category", "icon".
   - "intent": The detected intent.
Return ONLY the JSON object, wrapped in ```json\n...\n```.
"""
        messages = [{
            "role": "system",
            "content": prompt.format(
                customer_data=json.dumps(self.customer_data, indent=2),
                recent_transactions=json.dumps(self.recent_transactions, indent=2),
                travel_notice_data=json.dumps(self.travel_notice_data, indent=2),
                query=user_prompt
            )
        }]
        response, error = self._make_openai_request(messages)
        if error:
            logger.error("%s - OpenAI error: %s. Falling back to rule-based processing", self.__class__.__name__, error)
            self._add_analysis_step(f"OpenAI error: {error}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

        try:
            # Strip JSON code fences
            content = response.strip()
            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content[7:-4].strip()
            result = json.loads(content)
            self._add_decision_factor("openai_detected_intent", result.get("intent", "unknown"))
            self._add_analysis_step(f"OpenAI detected intent: {result.get('intent', 'unknown')}")
            self._set_response_construction("Response generated by OpenAI based on query analysis")
            for action in result.get("next_best_actions", []):
                self._add_next_best_action(
                    action["action"],
                    action["priority"],
                    action["description"],
                    action["category"],
                    action.get("icon", "🔹")
                )
            self._take_action("OpenAI Response Generated", f"Processed query: {user_prompt}")
            return {
                "response": result["response"],
                "reasoning_log": self.reasoning_log,
                "next_best_actions": result.get("next_best_actions", [])
            }
        except json.JSONDecodeError as e:
            logger.error("%s - Invalid JSON from OpenAI: %s. Falling back to rule-based", self.__class__.__name__, str(e))
            self._add_analysis_step(f"Invalid JSON from OpenAI: {str(e)}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

    def _process_rule_based(self, user_prompt: str) -> Dict:
        """Original rule-based processing logic."""
        self._add_analysis_step("Using rule-based card services processing")

        intent = self._determine_intent(user_prompt)
        self._add_decision_factor("determined_intent", intent)
        self._add_analysis_step(f"Determined user intent: {intent}")

        card_info = self._get_card_info()
        self._add_decision_factor("retrieved_card_info", card_info)
        self._add_analysis_step(f"Retrieved card info: Type={card_info.get('card_type')}, Last4={card_info.get('last_four')}, Status={card_info.get('status')}")

        inferred_card_issues = self._check_card_issues()
        self._add_decision_factor("inferred_card_issues", inferred_card_issues)
        self._add_analysis_step(f"Inferred potential card issues: {inferred_card_issues}")

        response = ""
        card_desc = f"your {card_info.get('card_type', 'card')} ending in {card_info.get('last_four', '****')}"

        if intent == "card_status":
            self._add_analysis_step("Handling 'card_status' intent.")
            self._consider_action("Provide card status information", "User explicitly asked about the card's status.")
            self._take_action("Card Status Report", f"Reported status of {card_desc} as {card_info.get('status', 'unknown')}.")
            self._set_response_construction("Providing information about current card status and any inferred issues.")

            response = f"Okay, I checked the status of {card_desc}. It is currently **{card_info.get('status', 'unknown')}**."

            if inferred_card_issues:
                response += "\n\nBased on recent activity, I also noticed a potential related issue:"
                critical_issue = next((issue for issue in inferred_card_issues if "lost" in issue or "fraud" in issue), inferred_card_issues[0])
                response += f"\n- {critical_issue}"

                if "reported lost" in card_info.get('status', ''):
                    response += "\n\nSince the card is marked as lost, it cannot be used. Would you like me to help you order a replacement?"
                    self._add_next_best_action(
                        "Order Replacement Card", "High",
                        "Initiate the process to order a replacement for the lost card.",
                        "Card Services", "🆕"
                    )
                elif "frozen" in card_info.get('status', '') or "blocked" in card_info.get('status', ''):
                    response += "\n\nBecause the card is blocked, transactions will be declined. We should resolve the reason for the block. Can I help with that?"
                    self._add_next_best_action(
                        "Resolve Card Block", "High",
                        "Investigate the reason for the card block and guide the user to resolve it.",
                        "Card Services", "🔓"
                    )
                elif "declined transactions" in critical_issue:
                    response += "\n\nHaving declined transactions can sometimes indicate an issue. Would you like to review those declines?"
                    self._add_next_best_action(
                        "Review Declined Transactions", "Medium",
                        "Offer to investigate the recent declined transactions.",
                        "Transaction Analysis", "📉"
                    )

        elif intent == "report_lost_stolen":
            self._add_analysis_step("Handling 'report_lost_stolen' intent.")
            self._consider_action("Mark card as lost/stolen", "User reported their card is lost or stolen.")

            if card_info.get('status', '').lower() == "reported lost" or card_info.get('status', '').lower() == "stolen":
                self._take_action("Lost/Stolen Report Skipped", f"Card {card_desc} was already marked as {card_info.get('status')}.")
                self._set_response_construction("Informing user card is already marked as lost/stolen and offering replacement.")
                response = f"I see {card_desc} is already marked as '{card_info.get('status')}' in our system. Because it's blocked, no further transactions can be made with it."
                response += "\n\nWould you like me to help you order a replacement card now?"
                self._add_next_best_action(
                    "Order Replacement Card", "High",
                    "Initiate the process to order a replacement for the already reported card.",
                    "Card Services", "🆕"
                )
            else:
                self._take_action("Report Card Lost/Stolen", f"Marked {card_desc} as lost/stolen in the system.")
                self._set_response_construction("Confirming card has been marked as lost/stolen, explaining consequences, and offering replacement.")
                card_info['status'] = 'reported lost'
                response = f"Okay, I've immediately marked {card_desc} as lost/stolen. For your security, this card is now **blocked** and cannot be used."
                response += "\n\nA replacement card will be automatically mailed to your address on file and should arrive in 5-7 business days. Is there anything else I can help with regarding this?"
                self._add_next_best_action(
                    "Verify Shipping Address", "Medium",
                    "Offer to confirm or update the shipping address for the replacement card.",
                    "Card Services", "🏠"
                )
                self._add_next_best_action(
                    "Offer Digital Card Access", "Medium",
                    "Inform user about options for immediate digital card access while waiting.",
                    "Digital Services", "📱"
                )
                self._add_next_best_action(
                    "Review Recent Transactions (Security)", "High",
                    "Suggest reviewing recent transactions for any unauthorized charges.",
                    "Security", "🛡️"
                )

        elif intent == "replace_card":
            self._add_analysis_step("Handling 'replace_card' intent (e.g., damaged card, expiring soon).")
            self._consider_action("Process card replacement", "User requested a replacement for their card (not necessarily lost/stolen).")
            self._take_action("Card Replacement Initiated", f"Initiated replacement process for {card_desc}.")
            self._set_response_construction("Confirming card replacement request and explaining the process.")

            expires_soon = False

            if expires_soon:
                response = f"Yes, I see {card_desc} is expiring soon. I've initiated the process to send you a new one. It should arrive within 5-7 business days."
            elif card_info.get('status', '').lower() == "active":
                response = f"Okay, I can process a replacement for {card_desc}. Your current card will remain active until you activate the new one."
                response += " The new card should arrive in 5-7 business days."
            else:
                response = f"Since {card_desc} is currently '{card_info.get('status')}', a replacement is typically part of resolving that status. We've already started that process if it was reported lost/stolen."
                response += " If it was blocked for another reason, let's resolve that first. Can I help with the reason it was blocked?"

            if card_info.get('status', '').lower() == "active" or expires_soon:
                self._add_next_best_action(
                    "Expedite Replacement Shipping", "Low",
                    "Offer expedited shipping options for the replacement card (may involve a fee).",
                    "Card Services", "🚀"
                )
                self._add_next_best_action(
                    "Update Card Design", "Low",
                    "If applicable, offer different card designs for the replacement.",
                    "Card Services", "🎨"
                )

        elif intent == "card_limits":
            self._add_analysis_step("Handling 'card_limits' intent.")
            self._consider_action("Provide card limit information", "User asked about spending or withdrawal limits.")
            self._take_action("Card Limits Report", f"Provided limit information for {card_desc}.")
            self._set_response_construction("Providing information about relevant card limits.")

            limits = self._get_card_limits(card_info)
            self._add_decision_factor("retrieved_card_limits", limits)

            response = f"Here are the current limits associated with {card_desc}:\n"
            response += f"- Daily Purchase Limit: ${limits.get('daily_purchase', 'N/A'):,}\n"
            response += f"- Daily ATM Withdrawal Limit: ${limits.get('daily_atm', 'N/A'):,}\n"
            if 'credit_limit' in limits:
                response += f"- Total Credit Limit: ${limits.get('credit_limit'):,}\n"
                response += f"- Available Credit: ${limits.get('available_credit'):,}"

            eligible_for_increase = card_info.get('eligible_for_credit_increase', False)
            self._add_decision_factor("eligible_for_limit_increase", eligible_for_increase)

            if eligible_for_increase and 'credit_limit' in limits:
                response += "\n\nGood news! You may be eligible for a credit limit increase. Would you like to explore that possibility?"
                self._add_next_best_action(
                    "Request Credit Limit Increase", "Medium",
                    "Offer to start the process for a credit limit increase.",
                    "Account Management", "📈"
                )
            elif not eligible_for_increase and 'credit_limit' in limits:
                response += "\n\nIf you need a higher limit in the future, feel free to ask, and we can review your account."

            self._add_next_best_action(
                "Set Spending Alerts", "Low",
                "Offer to set up alerts when spending approaches certain limits.",
                "Account Management", "🔔"
            )

        else:
            self._add_analysis_step("Handling 'general_inquiry' intent regarding the card.")
            self._consider_action("Provide general card information", "User asked a general question about their card.")
            self._take_action("General Card Information Provided", f"Provided overview for {card_desc}.")
            self._set_response_construction("Providing general information about the customer's card and highlighting any issues.")

            response = f"Let's talk about {card_desc}. It's a {card_info.get('card_type', '')} card, currently {card_info.get('status', 'active')}."

            if inferred_card_issues:
                response += "\n\nWhile checking, I did notice a potential issue based on recent activity:"
                issue = inferred_card_issues[0]
                response += f"\n- {issue}"
                response += "\n\nCan I help you look into this further?"
                if "lost" in issue:
                    self._add_next_best_action("Order Replacement Card", "High", "...", "Card Services", "🆕")
                elif "declined" in issue:
                    self._add_next_best_action("Review Declined Transactions", "Medium", "...", "Transaction Analysis", "📉")

            else:
                response += " Everything looks normal with the card right now. Do you have a specific question about its features, benefits, or something else?"
                self._add_next_best_action(
                    "Explore Card Benefits", "Low",
                    "Offer to explain the benefits and features associated with the card.",
                    "Card Services", "⭐"
                )
                self._add_next_best_action(
                    "Ask Specific Card Question", "Low",
                    "Prompt the user to ask their specific question about the card.",
                    "General Inquiry", "❓"
                )

        logger.info("%s completed rule-based processing", self.__class__.__name__)
        return {
            "response": response.strip(),
            "reasoning_log": self.reasoning_log,
            "next_best_actions": self.reasoning_log.get("next_best_actions", [])
        }

    def _determine_intent(self, user_prompt: str) -> str:
        """Determine the primary user intent regarding card services"""
        logger.debug("%s determining intent from prompt", self.__class__.__name__)
        prompt_lower = user_prompt.lower()

        if re.search(r"\b(lost|stolen|missing|can'?t find my card|someone took my card)\b", prompt_lower):
            self._add_analysis_step("Detected keywords related to lost or stolen card.")
            return "report_lost_stolen"

        if re.search(r"\b(replace|replacement|new card|damaged|broken|expired|expiring soon)\b", prompt_lower):
            if not re.search(r"\b(lost|stolen|missing)\b", prompt_lower):
                self._add_analysis_step("Detected keywords related to replacing a card (damaged, expiring, etc.).")
                return "replace_card"

        if re.search(r"\b(limit[s]?|spending limit|how much can i spend|withdrawal limit|max.*spend|max.*withdraw)\b", prompt_lower):
            self._add_analysis_step("Detected keywords related to card limits.")
            return "card_limits"

        if re.search(r"\b(status|active|inactive|frozen|blocked|is my card working)\b", prompt_lower):
            if not re.search(r"\b(replace|limit)\b", prompt_lower):
                self._add_analysis_step("Detected keywords related to card status.")
                return "card_status"

        self._add_analysis_step("No specific card service keywords found. Defaulting intent to 'general_inquiry'.")
        return "general_inquiry"

    def _get_card_info(self) -> Dict:
        """Simulate fetching basic card information"""
        logger.debug("%s fetching card info", self.__class__.__name__)
        self._add_analysis_step("Fetching basic card information (simulated).")

        card_info = {
            "card_type": self.customer_data.get("card_type", "Credit Card"),
            "last_four": self.customer_data.get("card_last_four", "7842"),
            "status": "active",
            "eligible_for_credit_increase": self.customer_data.get("eligible_for_upgrade", False),
            "rewards_program": self.customer_data.get("rewards_tier", "Standard")
        }

        for tx in self.recent_transactions:
            reason = tx.get("reason", "").lower()
            if "card reported lost" in reason or "stolen" in reason:
                card_info["status"] = "reported lost"
                self._add_analysis_step("Inferred card status changed to 'reported lost' based on transaction decline reason.")
                break

        return card_info

    def _check_card_issues(self) -> List[str]:
        """Infer potential issues from card status and transactions"""
        logger.debug("%s checking for card issues", self.__class__.__name__)
        issues = []
        card_status = self._get_card_info().get('status', 'unknown').lower()

        if card_status != "active":
            issues.append(f"The card is currently marked as '{card_status}'.")

        if card_status == "active":
            declined_transactions = [tx for tx in self.recent_transactions if tx.get("status", "").lower() == "declined"]
            if declined_transactions:
                issues.append(f"There {'has' if len(declined_transactions) == 1 else 'have'} been {len(declined_transactions)} declined transaction(s) recently while the card status is active.")

        if card_status == "active" and any("card reported lost" in tx.get("reason", "").lower() for tx in self.recent_transactions):
            issues.append("There's a discrepancy: the card is marked active, but a recent transaction was declined because the card was reported lost.")

        travel_notice_status = self.travel_notice_data.get("status", "Unknown").lower()
        if "error" in travel_notice_status or "not activated" in travel_notice_status:
            international_declines = []
            for tx in self.recent_transactions:
                if tx.get('status', '').lower() == 'declined':
                    location = tx.get('location', '').lower()
                    is_international = any(loc in location for loc in ["japan", "germany", "spain", "france", "italy", "uk"])
                    if is_international:
                        international_declines.append(tx)

            if international_declines:
                issues.append("An issue with the travel notice activation may be causing international declines.")

        return issues

    def _get_card_limits(self, card_info: Dict) -> Dict:
        """Simulate fetching card limits"""
        logger.debug("%s fetching card limits", self.__class__.__name__)
        self._add_analysis_step("Fetching card limits (simulated).")
        limits = {
            "daily_purchase": 5000,
            "daily_atm": 1000,
        }
        if "Credit" in card_info.get("card_type", ""):
            limits["credit_limit"] = self.customer_data.get("credit_limit", 10000)
            limits["available_credit"] = int(limits["credit_limit"] * 0.7)

        return limits


class GeneralInquiryAgent(BaseAgent):
    """Agent handling general account inquiries and information requests"""

    def process(self, user_prompt: str) -> Dict:
        logger.info("%s processing prompt: %s", self.__class__.__name__, user_prompt)
        self._add_analysis_step("Initializing GeneralInquiryAgent")
        self._add_analysis_step(f"Received user prompt: '{user_prompt}'")

        if self.openai_api_key and self.model:
            return self._process_with_openai(user_prompt)
        else:
            return self._process_rule_based(user_prompt)

    def _process_with_openai(self, user_prompt: str) -> Dict:
        """Process the prompt using OpenAI for intent detection and response generation."""
        self._add_analysis_step("Using OpenAI for general inquiry processing")
        prompt = """
                You are a banking general inquiry agent. Your task is to analyze a customer query about their account (e.g., balance, overview, contact preferences) and provide a response with recommended actions.

                **Context Data**:
                - Customer Info: {customer_data}
                - Recent Transactions: {recent_transactions}
                - Travel Notices: {travel_notice_data}
                - Customer Query: {query}

                **Instructions**:
                1. Analyze the query to identify the intent (e.g., check balance, get account overview, update contact preferences, general help).
                2. Use the customer data and transactions to provide relevant details.
                3. Generate a natural language response addressing the query.
                4. Suggest 1-3 next best actions with priority (High, Medium, Low), description, category, and an emoji icon.
                5. Return a JSON object with:
                - "response": The natural language response.
                - "next_best_actions": List of actions, each with "action", "priority", "description", "category", "icon".
                - "intent": The detected intent.
                Return ONLY the JSON object, wrapped in ```json\n...\n```.
                """
        messages = [{
            "role": "system",
            "content": prompt.format(
                customer_data=json.dumps(self.customer_data, indent=2),
                recent_transactions=json.dumps(self.recent_transactions, indent=2),
                travel_notice_data=json.dumps(self.travel_notice_data, indent=2),
                query=user_prompt
            )
        }]
        response, error = self._make_openai_request(messages)
        if error:
            logger.error("%s - OpenAI error: %s. Falling back to rule-based processing", self.__class__.__name__, error)
            self._add_analysis_step(f"OpenAI error: {error}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

        try:
            # Strip JSON code fences
            content = response.strip()
            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content[7:-4].strip()
            result = json.loads(content)
            self._add_decision_factor("openai_detected_intent", result.get("intent", "unknown"))
            self._add_analysis_step(f"OpenAI detected intent: {result.get('intent', 'unknown')}")
            self._set_response_construction("Response generated by OpenAI based on query analysis")
            for action in result.get("next_best_actions", []):
                self._add_next_best_action(
                    action["action"],
                    action["priority"],
                    action["description"],
                    action["category"],
                    action.get("icon", "🔹")
                )
            self._take_action("OpenAI Response Generated", f"Processed query: {user_prompt}")
            return {
                "response": result["response"],
                "reasoning_log": self.reasoning_log,
                "next_best_actions": result.get("next_best_actions", [])
            }
        except json.JSONDecodeError as e:
            logger.error("%s - Invalid JSON from OpenAI: %s. Falling back to rule-based", self.__class__.__name__, str(e))
            self._add_analysis_step(f"Invalid JSON from OpenAI: {str(e)}. Switching to rule-based processing")
            return self._process_rule_based(user_prompt)

    def _process_rule_based(self, user_prompt: str) -> Dict:
        """Original rule-based processing logic."""
        self._add_analysis_step("Using rule-based general inquiry processing")

        inquiry_type = self._determine_inquiry_type(user_prompt)
        self._add_decision_factor("determined_inquiry_type", inquiry_type)
        self._add_analysis_step(f"Determined inquiry type: {inquiry_type}")

        account_summary = self._gather_account_summary()
        self._add_decision_factor("account_summary_info", account_summary)
        self._add_analysis_step("Gathered account summary information.")

        response = ""
        customer_name = self.customer_data.get('name', 'Valued Customer')

        if inquiry_type == "account_overview":
            self._add_analysis_step("Handling 'account_overview' inquiry.")
            self._consider_action("Provide account overview", "User requested general account information or summary.")
            self._take_action("Account Overview Provided", "Generated summary of key account details.")
            self._set_response_construction("Creating a concise summary of the customer's account information and status.")

            response = f"Okay {customer_name}, here's a quick overview of your {account_summary.get('account_type', 'account')}:\n\n"
            response += f"- Account Holder: {account_summary.get('name', 'N/A')}\n"
            response += f"- Account Type: {account_summary.get('account_type', 'N/A')}\n"
            response += f"- Account Opened: {account_summary.get('account_opened', 'N/A')}\n"
            response += f"- Primary Card: {account_summary.get('card_type', 'N/A')} ending in {account_summary.get('card_last_four', '****')}\n"

            if account_summary.get("has_declined_transactions"):
                response += f"- Recent Activity Note: There have been some declined transactions recently.\n"
                self._add_next_best_action(
                    "Review Declined Transactions", "Medium",
                    "Offer to investigate the recent declined transactions.",
                    "Transaction Analysis", "📉"
                )
            if account_summary.get("has_travel_notice_issue"):
                response += f"- Travel Notice Note: There might be an issue with your current travel notice activation.\n"
                self._add_next_best_action(
                    "Check Travel Notice Status", "High",
                    "Offer to check and resolve issues with the travel notice.",
                    "Travel Services", "✈️"
                )

            response += "\nIs there anything specific in this overview you'd like to discuss further?"
            self._add_next_best_action(
                "Ask Follow-up Question", "Low",
                "Prompt the user if they have questions about the overview provided.",
                "General Inquiry", "❓"
            )

        elif inquiry_type == "balance_inquiry":
            self._add_analysis_step("Handling 'balance_inquiry'.")
            self._consider_action("Provide balance information", "User asked about account balance or funds.")
            self._take_action("Balance Information Provided", f"Provided average balance: {account_summary.get('average_balance', 'N/A')}.")
            self._set_response_construction("Providing balance information and context about recent activity.")

            balance = account_summary.get('average_balance', 'N/A')
            response = f"Your current average balance is approximately {balance}."

            recent_approved = [tx for tx in self.recent_transactions if tx.get("status", "").lower() == "approved"][:3]
            if recent_approved:
                response += "\n\nHere are a few of your most recent approved transactions:"
                for tx in recent_approved:
                    response += f"\n- {tx.get('date')}: {tx.get('amount')} at {tx.get('merchant')}"

            self._add_next_best_action(
                "See Full Transaction History", "Low",
                "Offer to show the complete recent transaction history.",
                "Account Management", "📜"
            )
            self._add_next_best_action(
                "Set Up Balance Alerts", "Low",
                "Suggest setting up notifications for low balance or large transactions.",
                "Account Management", "🔔"
            )

        elif inquiry_type == "contact_preferences":
            self._add_analysis_step("Handling 'contact_preferences' inquiry.")
            self._consider_action("Provide contact preference information", "User asked about communication settings.")
            self._take_action("Contact Preferences Reported", f"Provided current preference: {account_summary.get('contact_preference', 'N/A')}.")
            self._set_response_construction("Informing about current contact preferences and offering update options.")

            preference = account_summary.get('contact_preference', 'not set')
            email = account_summary.get('email', 'not provided')
            phone = account_summary.get('phone', 'not provided')

            response = f"Your current contact preference is set to **{preference}**."
            if preference.lower() == 'email':
                response += f" We have your email address as: {email}."
            elif preference.lower() == 'phone' or preference.lower() == 'sms':
                response += f" We have your phone number as: {phone}."
            else:
                response += f" We have your email as {email} and phone as {phone}."

            response += "\n\nWould you like to update your preferred contact method or change the email/phone number we have on file?"

            self._add_next_best_action(
                "Update Contact Method", "Medium",
                "Offer to change the preferred method (Email, SMS, Phone, Mail).",
                "Account Management", "⚙️"
            )
            self._add_next_best_action(
                "Update Contact Details", "Medium",
                "Offer to update the email address or phone number on file.",
                "Account Management", "✏️"
            )

        else:
            self._add_analysis_step("Handling 'general_help' or unclear inquiry.")
            self._consider_action("Provide general assistance options", "User asked for help or the inquiry was not specific.")
            self._take_action("General Help Options Provided", "Listed common tasks the agent can perform.")
            self._set_response_construction("Creating a helpful starting point, listing capabilities, and offering personalized suggestions.")

            response = f"Hello {customer_name}! I can help with various banking tasks. For example, I can assist you with:\n\n"
            response += "- Checking your account balance or recent transactions\n"
            response += "- Managing travel notices for your trips\n"
            response += "- Card services like reporting lost/stolen cards or checking limits\n"
            response += "- Updating your contact preferences\n"

            response += "\nHow can I help you specifically today?"

            if account_summary.get("has_declined_transactions"):
                self._add_next_best_action(
                    "Discuss Recent Declines", "Medium",
                    "Offer to look into the recent declined transactions.",
                    "Transaction Analysis", "📉"
                )
            elif account_summary.get("has_travel_notice_issue"):
                self._add_next_best_action(
                    "Resolve Travel Notice Issue", "High",
                    "Offer to fix the identified issue with the travel notice.",
                    "Travel Services", "✈️"
                )
            elif account_summary.get("card_status") != "active":
                self._add_next_best_action(
                    f"Address Card Status ({account_summary.get('card_status')})", "High",
                    f"Offer to help resolve the issue with the card being {account_summary.get('card_status')}.",
                    "Card Services", "💳"
                )
            else:
                self._add_next_best_action(
                    "Review Account Security", "Low",
                    "Offer to review security settings or recent login activity.",
                    "Security", "🔒"
                )

        logger.info("%s completed rule-based processing", self.__class__.__name__)
        return {
            "response": response.strip(),
            "reasoning_log": self.reasoning_log,
            "next_best_actions": self.reasoning_log.get("next_best_actions", [])
        }

    def _determine_inquiry_type(self, user_prompt: str) -> str:
        """Determine the type of general inquiry based on keywords"""
        logger.debug("%s determining inquiry type from prompt", self.__class__.__name__)
        prompt_lower = user_prompt.lower()

        if re.search(r"\b(balance|how much.*in my account|funds available|account total)\b", prompt_lower):
            self._add_analysis_step("Detected keywords related to balance inquiry.")
            return "balance_inquiry"

        if re.search(r"\b(contact|email|phone|text|sms|notification|preferences|how.*contact me)\b", prompt_lower):
            if "travel notice" not in prompt_lower and "transaction" not in prompt_lower:
                self._add_analysis_step("Detected keywords related to contact preferences.")
                return "contact_preferences"

        if re.search(r"\b(overview|summary|account details|my account|information.*account)\b", prompt_lower):
            if not re.search(r"\b(balance|contact|travel|transaction|card)\b", prompt_lower):
                self._add_analysis_step("Detected keywords related to account overview.")
                return "account_overview"

        if re.search(r"\b(help|support|question|assist|can you)\b", prompt_lower) or \
           not any(re.search(r"\b(balance|contact|overview|travel|transaction|card)\b", prompt_lower)):
            self._add_analysis_step("Inquiry seems general or asking for help. Defaulting to 'general_help'.")
            return "general_help"

        self._add_analysis_step("Could not determine specific inquiry type. Defaulting to 'general_help'.")
        return "general_help"

    def _gather_account_summary(self) -> Dict:
        """Gather key account details and flags for enriching responses"""
        logger.debug("%s gathering account summary", self.__class__.__name__)
        self._add_analysis_step("Gathering summary details from customer data, travel notice, and transactions.")
        summary = {
            "name": self.customer_data.get("name", "N/A"),
            "account_type": self.customer_data.get("account_type", "N/A"),
            "account_opened": self.customer_data.get("account_opened", "N/A"),
            "average_balance": self.customer_data.get("average_balance", "N/A"),
            "card_type": self.customer_data.get("card_type", "N/A"),
            "card_last_four": self.customer_data.get("card_last_four", "****"),
            "contact_preference": self.customer_data.get("contact_preference", "N/A"),
            "email": self.customer_data.get("email", "N/A"),
            "phone": self.customer_data.get("phone", "N/A"),
            "has_declined_transactions": any(tx.get("status", "").lower() == "declined" for tx in self.recent_transactions),
            "has_travel_notice_issue": "error" in self.travel_notice_data.get("status", "").lower() or \
                                      "not activated" in self.travel_notice_data.get("status", "").lower(),
            "card_status": "active"
        }

        for tx in self.recent_transactions:
            reason = tx.get("reason", "").lower()
            if "card reported lost" in reason or "stolen" in reason:
                summary["card_status"] = "reported lost"
                break

        return summary


def get_agent_for_routing(agent_name: str, customer_data: Dict, travel_notice_data: Dict, recent_transactions: List[Dict], openai_api_key: Optional[str] = None, model: Optional[str] = None):
    """Factory function to create the appropriate agent based on routing decision"""
    logger.info("Creating agent: %s", agent_name)
    agent_map = {
        "TransactionAnalysisAgent": TransactionAnalysisAgent,
        "TravelNoticeAgent": TravelNoticeAgent,
        "CardServicesAgent": CardServicesAgent,
        "GeneralInquiryAgent": GeneralInquiryAgent
    }

    agent_class = agent_map.get(agent_name, GeneralInquiryAgent)
    logger.debug("Selected agent class: %s", agent_class.__name__)
    return agent_class(customer_data, travel_notice_data, recent_transactions, openai_api_key, model)