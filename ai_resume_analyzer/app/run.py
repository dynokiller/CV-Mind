import subprocess
import sys

print("Starting main app...")
subprocess.Popen([sys.executable, "app.py"])

print("Starting mail service...")
subprocess.Popen([sys.executable, "mail_service/index.py"])

input("Both services are running. Press ENTER to stop...\n")
