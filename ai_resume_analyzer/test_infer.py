
import sys
import json
import logging
logging.getLogger("xgboost").setLevel(logging.ERROR)

from inference.inference import predict_domain

resume_text = """
Experienced Cyber Security Analyst with a strong background in penetration testing, network security, and vulnerability assessments. 
Certified Ethical Hacker (CEH) and CISSP. Proficient in Kali Linux, Wireshark, Metasploit, and SIEM tools like Splunk. 
Managed firewalls, intrusion detection systems (IDS), and conducted regular security audits.
"""

res = predict_domain(resume_text)
print("PREDICTED DOMAIN:", res["predicted_domain"])
print("CONFIDENCE:", res["confidence"])

