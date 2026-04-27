
from email.message import EmailMessage
from datetime import datetime
import os

MAIL_EMAIL = os.getenv("MAIL_EMAIL")
MAIL_FROM_NAME = os.getenv("MAIL_FROM_NAME", "Resume Analyzer")

if not MAIL_EMAIL:
    raise RuntimeError("MAIL_EMAIL missing")

def build_reset_email(data: dict) -> EmailMessage:
    email = data["email"]
    reset_link = data["redirect"]  #
    
    msg = EmailMessage()
    msg["Subject"] = "Reset Your Resume Analyzer Password"
    msg["From"] = f"{MAIL_FROM_NAME} <{MAIL_EMAIL}>"
    msg["To"] = email

    html_body = f"""
                  <!DOCTYPE html>
                    <html>
                    <head>
                      <meta charset="UTF-8">
                      <meta name="viewport" content="width=device-width, initial-scale=1.0">
                      <style>
                        body {{
                          margin: 0;
                          padding: 0;
                          background: #f4f6ff;
                          font-family: Arial, sans-serif;
                        }}
                        .wrapper {{
                          width: 100%;
                          padding: 30px 12px;
                          background: #f4f6ff;
                        }}
                        .card {{
                          max-width: 560px;
                          margin: auto;
                          background: white;
                          border-radius: 18px;
                          overflow: hidden;
                          box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
                          border: 1px solid rgba(99, 102, 241, 0.15);
                        }}
                        .header {{
                          padding: 22px 20px;
                          text-align: center;
                          color: white;
                          background: linear-gradient(135deg, #2f6fff, #7c3aed);
                        }}
                        .header h1 {{
                          margin: 0;
                          font-size: 22px;
                          font-weight: 800;
                          letter-spacing: 0.4px;
                        }}
                        .content {{
                          padding: 22px 20px;
                          color: #0f172a;
                          text-align: center;
                        }}
                        .content p {{
                          margin: 0 0 14px;
                          color: #475569;
                          font-size: 14px;
                          line-height: 1.6;
                          font-weight: 600;
                        }}
                        .otp-box {{
                          margin: 18px auto;
                          width: fit-content;
                          padding: 14px 26px;
                          border-radius: 14px;
                          font-size: 28px;
                          font-weight: 900;
                          letter-spacing: 6px;
                          color: #2f6fff;
                          background: rgba(47, 111, 255, 0.12);
                          border: 1px dashed rgba(124, 58, 237, 0.5);
                        }}
                        .note {{
                          font-size: 12px;
                          color: #64748b;
                          margin-top: 18px;
                          font-weight: 700;
                        }}
                        .footer {{
                          padding: 14px 20px;
                          text-align: center;
                          font-size: 12px;
                          font-weight: 700;
                          color: #94a3b8;
                          background: #f8fafc;
                          border-top: 1px solid rgba(15, 23, 42, 0.08);
                        }}
                      </style>
                    </head>

                    <body>
                      <div class="wrapper">
                        <div class="card">
                          <div class="header">
                            <h1>Resume Analyzer - OTP Verification</h1>
                          </div>

                          <div class="content">
                            <p>Hello, </p>
                            <p>Use the OTP below to verify your account.</p>

                            <a href="{reset_link}" class="btn">Reset Password</a>

                            <p class="note">
                              This link is valid for <b>15 minutes</b>.<br/>
                              If you didn't request this, you can safely ignore this email.
                            </p>
                          </div>

                          <div class="footer">
                            © {datetime.now().year} Resume Analyzer | Cloud Infrastructure Project
                          </div>
                        </div>
                      </div>
                    </body>
                  </html>
                """
    msg.add_alternative(html_body, subtype="html")
    return msg
