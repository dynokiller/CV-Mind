import requests, re

s = requests.Session()
s.get('http://127.0.0.1:5000/dev-login', allow_redirects=True)
print('Logged in. Uploading resume...')

with open(r'C:\Users\Admin\Desktop\test_resume.txt', 'rb') as f:
    files = {'resume': ('test_resume.txt', f, 'text/plain')}
    r = s.post('http://127.0.0.1:5000/upload-resume', files=files,
               data={'job_description': ''}, timeout=60, allow_redirects=True)

print('Status:', r.status_code, '| Final URL:', r.url)
text = r.text

if 'Latest Analysis Result' in text:
    print('SUCCESS: Analysis result card shown on dashboard!')
    dm = re.search(r'PREDICTED DOMAIN.{0,100}status success[^>]*>([^<]+)<', text, re.DOTALL)
    cf = re.search(r'CONFIDENCE.{0,200}([\d.]+)%', text, re.DOTALL)
    sc = re.search(r'RESUME SCORE.{0,200}>([\d]+)<span', text, re.DOTALL)
    print('  Domain:', dm.group(1).strip() if dm else 'N/A')
    print('  Confidence:', cf.group(1)+'%' if cf else 'N/A')
    print('  Score:', sc.group(1)+'/100' if sc else 'N/A')
else:
    print('No result card — checking status...')
    statuses = re.findall(r'class="status (\w+)"', text)
    print('  Activity statuses:', statuses)
    # Print a chunk of the page for debugging
    idx = text.find('Recent Activity')
    if idx > 0:
        print('  Page excerpt:', text[idx:idx+500])
