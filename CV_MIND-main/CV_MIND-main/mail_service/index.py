import os
import smtplib
from email.message import EmailMessage
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from handlers import HANDLERS

load_dotenv()
app = Flask(__name__)

MAIL_EMAIL = os.getenv("MAIL_EMAIL")
MAIL_APP_PASSWORD = os.getenv("MAIL_APP_PASSWORD")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

def send_email(msg: EmailMessage) -> bool:
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(MAIL_EMAIL, MAIL_APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print("SMTP Error:", e)
        return False


@app.route("/mail_service/index", methods=["POST"])
def mail_dispatcher():
    data = request.get_json() or {}

    email = data.get("email")
    mail_type = data.get("type")

    if not email or not mail_type:
        return jsonify({
            "success": False,
            "message": "email and type required"
        }), 400

    handler = HANDLERS.get(mail_type)
    if not handler:
        return jsonify({
            "success": False,
            "message": f"Invalid mail type: {mail_type}"
        }), 400

    try:
        msg = handler(data)

        if send_email(msg):
            return jsonify({"success": True}), 200

        return jsonify({
            "success": False,
            "message": "Email sending failed"
        }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
