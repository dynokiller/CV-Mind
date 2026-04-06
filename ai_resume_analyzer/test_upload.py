import requests
import os

# Configuration
API_URL = "http://127.0.0.1:8000/analyze"
RESUME_PATH = "sample_resume.pdf"  # Replace with your actual resume path
JOB_DESCRIPTION = """
We are looking for a Machine Learning Engineer with experience in Python, Scikit-learn, and NLP.
The candidate should have strong problem-solving skills and experience with FastAPI.
"""

def test_api():
    # 1. Check if resume exists
    if not os.path.exists(RESUME_PATH):
        print(f"Error: {RESUME_PATH} not found. Please create or place a resume file here.")
        # Create a dummy text resume for testing if PDF doesn't exist
        with open("sample_resume.txt", "w") as f:
            f.write("John Doe\nPython Developer\nSkills: Python, FastAPI, Machine Learning\nEmail: john@example.com")
        print("Created sample_resume.txt for testing.")
        resume_file = "sample_resume.txt"
    else:
        resume_file = RESUME_PATH

    # 2. Prepare payload
    files = {'resume': open(resume_file, 'rb')}
    data = {'job_description': JOB_DESCRIPTION}

    # 3. Send Request
    print(f"Sending request to {API_URL}...")
    try:
        response = requests.post(API_URL, files=files, data=data)
        
        # 4. Print Response
        if response.status_code == 200:
            print("\n✅ Success! Analysis Result:")
            print(response.json())
        else:
            print(f"\n❌ Error {response.status_code}:")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        print("Make sure the server is running: uvicorn app.main:app --reload")

if __name__ == "__main__":
    test_api()
