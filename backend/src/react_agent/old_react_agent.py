# import json
# import sys
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from loguru import logger
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from typing import List, Dict

# # Configure logging
# logger.remove()
# logger.add(sys.stdout, level="INFO")

# # System Prompt
# system_prompt = """
# You are Shahzad. You are a friendly and neutral entity developed by AI to help provide users with 
# their tasks and questions. Your job is to choose the best possible function to carry out the user question.
# Select final_answer if you do not need to call external functions or have already successfully 
# called all necessary functions. 
# """

# # Function to send an email
# def send_email_smtp(to, subject, body):
#     try:
#         smtp_server = "smtp.gmail.com"
#         smtp_port = 587
#         sender_email = "aicoding11@gmail.com"
#         sender_password = "ikex...."

#         message = MIMEMultipart()
#         message["From"] = sender_email
#         message["To"] = to
#         message["Subject"] = subject
#         message.attach(MIMEText(body, "plain"))

#         with smtplib.SMTP(smtp_server, smtp_port) as server:
#             server.starttls()
#             server.login(sender_email, sender_password)
#             server.sendmail(sender_email, to, message.as_string())

#         logger.info(f"Email successfully sent to {to}")
#         return {"status": "success", "message": f"Email sent to {to}"}
#     except Exception as e:
#         logger.error(f"Failed to send email: {e}")
#         return {"status": "error", "message": f"Failed to send email: {e}"}


# # Tool class
# class Tool:
#     def __init__(self, description, endpoint=None):
#         self.description = description
#         self.endpoint = endpoint

#     def api_endpoint(self, arguments=None):
#         if self.endpoint == "send_email_smtp":
#             return send_email_smtp(**arguments)
#         return {"status": "success", "result": f"Called endpoint: {self.endpoint}, with args: {arguments}"}


# # Tools
# final_answer_description = {
#     "name": "final_answer",
#     "description": "Responds to the user with the final answer or asks for more information.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "message": {"type": "string", "description": "The response message"}
#         },
#         "required": ["message"],
#     },
# }
# final_answer = Tool(final_answer_description)

# send_email_gmail_description = {
#     "name": "send_email_gmail",
#     "description": "Send an email via SMTP.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "to": {"type": "string", "description": "Recipient's email address."},
#             "subject": {"type": "string", "description": "Subject of the email."},
#             "body": {"type": "string", "description": "Content of the email."},
#         },
#         "required": ["to", "subject", "body"],
#     },
# }
# send_email_gmail = Tool(send_email_gmail_description, "send_email_smtp")

# tool_instances = {
#     "send_email_gmail": send_email_gmail,
#     "final_answer": final_answer,
# }

# # LLM Setup
# llm = "EleutherAI/gpt-neo-1.3B"
# tokenizer = AutoTokenizer.from_pretrained(llm)
# model = AutoModelForCausalLM.from_pretrained(llm)
# tokenizer.pad_token = tokenizer.eos_token
# text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, truncation=True)

# # Function to sanitize JSON response
# def sanitize_json_response(response: str) -> str:
#     """
#     Extracts the first valid JSON object from a string and returns it as a JSON-compatible string.
#     """
#     import re
#     try:
#         # This pattern looks for the first occurrence of a JSON object
#         json_pattern = r"{(?:[^{}]|(?:\".*?\"))*}"
#         match = re.search(json_pattern, response)
#         if match:
#             valid_json = match.group()
#             # Ensure the extracted JSON is valid
#             json.loads(valid_json)
#             logger.info(f"Sanitized JSON string: {valid_json}")
#             return valid_json
#         logger.warning("No valid JSON found in response.")
#         return "{}"
#     except (json.JSONDecodeError, Exception) as e:
#         logger.error(f"Error during JSON sanitization: {e}")
#         return "{}"


# # Function to select action
# def select_action(messages):
#     action_prompt = """
#     Based on the user input and context, decide on the appropriate action:
#     - Use "send_email_gmail" if the user wants to send an email.
#     - Use "final_answer" if no further actions are needed or clarification is required.

#     Respond with a JSON object containing:
#     - "thought": A brief explanation of the decision.
#     - "action": The selected action name ("send_email_gmail" or "final_answer").
#     """
#     prompt = f"{action_prompt}\nMessages: {json.dumps(messages)}\nSelect an action:"
#     response = generate_response(prompt)
#     sanitized_response = sanitize_json_response(response)
#     try:
#         return json.loads(sanitized_response)
#     except json.JSONDecodeError:
#         logger.error(f"Error decoding JSON response: {sanitized_response}")
#         return {"thought": "Error in parsing JSON response.", "action": "final_answer"}

# # Function to generate response
# def generate_response(prompt):
#     result = text_generator(prompt, truncation=True)
#     try:
#         return result[0]['generated_text']
#     except (IndexError, KeyError):
#         logger.error("Failed to retrieve generated text from AI response.")
#         return "{}"

# # Function to get action arguments
# def get_action_arguments(action_name, messages, description):
#     prompt = f"Action: {action_name}\nMessages: {json.dumps(messages)}\nDescription: {json.dumps(description)}\nProvide action arguments:"
#     response = generate_response(prompt)
#     sanitized_response = sanitize_json_response(response)
#     try:
#         return json.loads(sanitized_response)
#     except json.JSONDecodeError:
#         logger.error(f"Error decoding JSON response: {sanitized_response}")
#         if action_name == "final_answer":
#             return {"message": "Please provide the subject, body, and recipient's email address."}
#         return {}

# # Function to run action
# def run_action(tool, arguments=None):
#     try:
#         if all(key in arguments for key in ['to', 'subject', 'body']):
#             return tool.api_endpoint(arguments)
#         return {"status": "error", "message": "Missing 'to', 'subject', or 'body' for sending email."}
#     except Exception as e:
#         logger.error(f"Error running action: {e}")
#         return {"status": "error", "message": str(e)}

# # React agent to process user input
# def react_agent(user_prompt: str, past_messages: List[Dict[str, Dict[str, str]]] = []):
#     try:
#         messages = [{"role": "system", "content": system_prompt}]
#         if past_messages:
#             messages += past_messages[-4:]
#         messages.append({"role": "user", "content": user_prompt})

#         response = {"data": [], "res": None}
#         while True:
#             action = select_action(messages)
#             thought = action.get("thought", "Default thought for this action.")
#             function_name = action.get("action", "final_answer")

#             if function_name not in tool_instances:
#                 thought = "Invalid action selected. Providing a final response."
#                 function_name = "final_answer"

#             messages.append({"role": "assistant", "content": thought})
#             if function_name in tool_instances:
#                 tool = tool_instances[function_name]
#                 arguments = get_action_arguments(function_name, messages, tool.description)

#                 if any(arg not in arguments or not arguments[arg] for arg in ["to", "subject", "body"]):
#                     for arg in ["to", "subject", "body"]:
#                         if arg not in arguments or not arguments[arg]:
#                             messages.append({"role": "assistant", "content": f"Please provide the {arg}."})
#                     response["res"] = "Email information is incomplete. Requested additional details."
#                     break

#                 result = run_action(tool, arguments)
#                 response["data"].append(result)
#                 messages.append({"role": "function", "name": function_name, "content": json.dumps(result)})

#             if function_name == "final_answer":
#                 response["res"] = arguments.get("message", "Action completed successfully.")
#                 break

#         return response

#     except Exception as e:
#         logger.error(f"Error in react_agent: {e}")
#         return {"status": "error", "message": str(e)}

# # Example usage
# if __name__ == "__main__":
#     user_prompt = "Send an email to shahzad20022002@gmail.com with subject 'Hello' and body 'How are you?'"
#     response = react_agent(user_prompt)
#     print(response)
