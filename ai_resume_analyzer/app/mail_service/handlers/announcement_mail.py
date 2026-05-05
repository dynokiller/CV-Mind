import os
from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MAIL_EMAIL = os.getenv("MAIL_EMAIL")
MAIL_FROM_NAME = os.getenv("MAIL_FROM_NAME", "Resume Analyzer Team")

def build_announcement_email(data):
    email = data["email"]
    subject = data.get("subject", "System Announcement")
    message = data.get("message", "")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{MAIL_FROM_NAME} <{MAIL_EMAIL}>"
    msg["To"] = email

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Inter', Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 0; }}
            .container {{ max-width: 600px; margin: 20px auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 30px; text-align: center; color: white; }}
            .content {{ padding: 30px; color: #1e293b; line-height: 1.6; }}
            .footer {{ background: #f1f5f9; padding: 20px; text-align: center; color: #64748b; font-size: 12px; }}
            .button {{ display: inline-block; padding: 12px 24px; background: #6366f1; color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{subject}</h1>
            </div>
            <div class="content">
                <p>{message.replace('\n', '<br>')}</p>
                <a href="https://cv-mind.vercel.app" class="button">Visit Dashboard</a>
            </div>
            <div class="footer">
                &copy; {datetime.now().year} CV Mind. All rights reserved.
            </div>
        </div>
    </body>
    </html>
    """
    msg.add_alternative(html_body, subtype="html")
    return msg
