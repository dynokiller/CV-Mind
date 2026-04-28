import re
import os

with open('app/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add smtplib to imports if not there
if "import smtplib" not in content:
    content = content.replace("import sys, os, time", "import sys, os, time, smtplib")

new_send_mail = """def send_mail(email: str, mail_type: str, otp: str = None, redirect: str = None):
    try:
        from mail_service.handlers.otp_mail import build_otp_email
        from mail_service.handlers.reset_mail import build_reset_email
        
        MAIL_EMAIL = os.getenv("MAIL_EMAIL")
        MAIL_APP_PASSWORD = os.getenv("MAIL_APP_PASSWORD")
        SMTP_HOST = "smtp.gmail.com"
        SMTP_PORT = 465
        
        if not MAIL_EMAIL or not MAIL_APP_PASSWORD:
            print("[ERROR] MAIL_EMAIL or MAIL_APP_PASSWORD not set in .env")
            return {"success": False, "message": "Mail service not configured"}
            
        data = {"email": email, "type": mail_type, "otp": otp, "redirect": redirect}
        
        if mail_type == "OTP_VERIFY":
            msg = build_otp_email(data)
        elif mail_type == "RESET_PASSWORD":
            msg = build_reset_email(data)
        else:
            return {"success": False, "message": f"Invalid mail type: {mail_type}"}
            
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(MAIL_EMAIL, MAIL_APP_PASSWORD)
            smtp.send_message(msg)
            
        return {"success": True, "message": "Mail sent successfully"}
        
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return {"success": False, "message": "Mail service unreachable"}
"""

pattern2 = re.compile(r'def send_mail\(.*?\):.*?(?=\n# -+)', re.DOTALL)

if "def send_mail(" in content:
    content = pattern2.sub(new_send_mail, content, count=1)

with open('app/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
