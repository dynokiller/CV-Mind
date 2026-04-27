import requests

s = requests.Session()
s.get('http://127.0.0.1:5000/dev-login', allow_redirects=True)
print('Logged in.')

# Step 1: Test ML server directly
print('\n--- Testing ML server directly ---')
with open(r'C:\Users\Admin\Desktop\test_resume.txt', 'rb') as f:
    r_ml = requests.post(
        'http://127.0.0.1:8080/analyze_resume',
        files={'file': ('test_resume.txt', f, 'text/plain')},
        timeout=30
    )
print('ML Status:', r_ml.status_code)
if r_ml.status_code == 200:
    data = r_ml.json()
    print('  predicted_domain:', data.get('predicted_domain'))
    print('  confidence:', data.get('confidence'))
    strength = data.get('resume_strength', {})
    print('  resume_strength.score:', strength.get('score'))
    print('  resume_strength.strengths:', strength.get('strengths', [])[:2])
    print('  resume_strength.improvements:', strength.get('improvements', [])[:2])
    print('  top_keywords:', [k['word'] for k in data.get('top_keywords', [])[:5]])
    print('  latency_ms:', data.get('latency_ms'))
else:
    print('  Error:', r_ml.status_code, r_ml.text[:300])

# Step 2: Test Flask upload endpoint
print('\n--- Testing Flask upload-resume ---')
with open(r'C:\Users\Admin\Desktop\test_resume.txt', 'rb') as f:
    r_flask = s.post(
        'http://127.0.0.1:5000/upload-resume',
        files={'resume': ('test_resume.txt', f, 'text/plain')},
        data={'job_description': ''},
        timeout=60,
        allow_redirects=False  # Don't follow redirect so we can see what happens
    )
print('Flask Status:', r_flask.status_code)
print('Flask Location:', r_flask.headers.get('Location', 'No redirect'))
if r_flask.status_code in (200, 302):
    print('Redirect target:', r_flask.headers.get('Location'))
else:
    print('Response:', r_flask.text[:400])
