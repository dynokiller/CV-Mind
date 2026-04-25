import json
from resume_analyzer import ResumeAnalyzer

print("Loading Intelligent Analyzer...")
analyzer = ResumeAnalyzer()

sample_hr = "Experienced Human Resources professional with 8 years in corporate recruitment, onboarding, and employee relations. Managed payroll processes and talent acquisition for multiple tech startups. Strong knowledge of labor laws and performance management strategies."

sample_cyber = "Security Operations Center (SOC) Analyst Level 2. I have hands-on experience using Splunk for threat hunting and incident response. Certified in CISSP and familiar with network penetration testing and analyzing malware."

sample_student = "Recent college graduate with a degree in Communications. I was the president of the debate club and organized several fundraising events. I am highly motivated, a quick learner, and excellent at public speaking and relationship building."

resumes = {
    "HR Manager": sample_hr,
    "Cyber Security Analyst": sample_cyber,
    "Recent Grad (Ambiguous)": sample_student
}

for title, text in resumes.items():
    print(f"\n\n=======================================")
    print(f"--- Testing {title} ---")
    print(f"Input: {text}")
    print(f"=======================================")
    result = analyzer.analyze(text)
    print(json.dumps(result, indent=2))
