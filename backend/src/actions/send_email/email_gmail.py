import re
from loguru import logger
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Environment Variables for SMTP
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
# Email Sending Function
def is_valid_email(email):
    """
    Validates an email address using a regex.
    """
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(email_regex, email) is not None

def send_email(to: str, subject: str, body: str):
    """
    Sends an email with the provided parameters (to, subject, body).
    """
    try:
        if not is_valid_email(to):
            raise ValueError(f"Invalid email address: {to}")
        
        logger.info(f"Sending email to {to} with subject '{subject}'...")
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to, msg.as_string())

        logger.info(f"Email sent successfully to {to}.")
        return {"status": "success", "message": f"Email sent successfully to {to}."}
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return {"status": "error", "message": str(e)}