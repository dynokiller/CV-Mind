import requests

MAIL_SERVICE_URL = "http://127.0.0.1:8000/api/index"

email = input("Enter email: ").strip()
otp = input("Enter OTP: ").strip()

payload = {
    "email": email,
    "type": "OTP_VERIFY",
    "otp": otp
}

response = requests.post(MAIL_SERVICE_URL, json=payload)

print("\nStatus Code:", response.status_code)
print("Response:", response.json())
