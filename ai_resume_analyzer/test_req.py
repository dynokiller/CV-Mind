
import requests
import json

resume_text = "Experienced Cyber Security Analyst with a strong background in penetration testing, network security, and vulnerability assessments. Certified Ethical Hacker (CEH) and CISSP. Proficient in Kali Linux, Wireshark, Metasploit, and SIEM tools."

response = requests.post("http://127.0.0.1:8000/analyze", json={
    "resume_text": resume_text,
    "job_description": "We need a cyber security person."
})

print("STATUS:", response.status_code)
try:
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", e, response.text)

