
from reportlab.pdfgen import canvas
text = "Experienced Cyber Security Analyst with a strong background in penetration testing, network security, and vulnerability assessments. Certified Ethical Hacker (CEH) and CISSP. Proficient in Kali Linux, Wireshark, Metasploit, and SIEM tools like Splunk. Managed firewalls, intrusion detection systems (IDS), and conducted regular security audits."
c = canvas.Canvas("test_resume.pdf")
c.drawString(100, 750, text)
c.save()

