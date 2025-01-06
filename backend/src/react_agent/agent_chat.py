from loguru import logger
from src.llm_monarch.llm import text_generator
import re
import json
from src.actions.send_email.email_gmail import send_email

# LLM Prompt Interpretation
def interpret_prompt(prompt: str):
    """
    Uses the LLM to interpret the user's intent and extract action details.
    """
    logger.info(f"Interpreting prompt: {prompt}")
    response = text_generator(
        f"""
        You are a JSON-based assistant that extracts actions and parameters from user instructions.
        Your output should ONLY contain JSON. No additional text or explanation.

        Example input: "Send an email to test@example.com with subject 'Hello' and body 'How are you?'"
        Example output:
        {{
            "action": "send_email",
            "parameters": {{
                "to": "test@example.com",
                "subject": "Hello",
                "body": "How are you?"
            }}
        }}

        Now, process this input:
        "{prompt}"
        """
    )[0]["generated_text"]

    logger.info(f"LLM Response: {response}")
    
    # Directly match and extract the JSON content from the LLM response
    json_match = re.search(r'(\{.*\})', response, re.DOTALL)
    if json_match:
        json_data = json_match.group(1)
        try:
            parsed_data = json.loads(json_data)
            
            # Validate the action name
            if parsed_data.get("action") == "action_name":
                raise ValueError("Action name not properly extracted.")
            
            return parsed_data
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON from LLM response.")
    else:
        raise ValueError("No valid JSON found in LLM response.")

# Execute Action Based on LLM Output
def execute_action(action_data: dict, user_prompt: str):
    """
    Executes the specified action based on the parsed JSON data.
    """
    try:
        action = action_data.get("action")
        parameters = action_data.get("parameters", {})

        logger.info(f"Action: {action}")
        logger.info(f"Parameters: {parameters}")

        # Check if the email in the parameters is still a placeholder and correct it
        if parameters.get("to") == "test@example.com":
            # Extract email from the user prompt
            email_match = re.search(r"to\s([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", user_prompt)
            if email_match:
                parameters["to"] = email_match.group(1)

        if action == "send_email":
            return send_email(**parameters)
        elif action == "final_answer":
            return {"status": "success", "message": parameters.get("message", "No message provided.")}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    except Exception as e:
        logger.error(f"Failed to execute action: {e}")
        return {"status": "error", "message": str(e)}