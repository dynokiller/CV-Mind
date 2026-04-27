from flask import Flask, render_template, redirect, url_for, request, session, flash, make_response, jsonify
# Database collections replaced with in-memory lists (stateless)
user_collection = []
stats_collection = []
activity_collection = []
file_integrity_collection = []
reset_tokens_collection = []
from werkzeug.middleware.proxy_fix import ProxyFix
from itsdangerous import URLSafeTimedSerializer
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
import os, time, requests, pytz, random, re, uuid
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
import hashlib
import secrets

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "fallbacksecretkey"

# Fix for Vercel/Proxy (solves Google OAuth CSRF issues)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Security headers for session cookies
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)


# ----------------------------
# Upload Config
# ----------------------------

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "png", "jpg", "jpeg"}
MAX_FILE_SIZE = 28 * 1024 * 1024  # 28MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



# Timezone
IST = pytz.timezone("Asia/Kolkata")
app.permanent_session_lifetime = timedelta(minutes=20)


# ----------------------------
# OAuth Setup
# ----------------------------
oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Fallback secret key for itsdangerous
serializer = URLSafeTimedSerializer(
    app.secret_key if app.secret_key else "fallbacksecretkey"
)

# ----------------------------
# Helpers
# ----------------------------


def safe_find_one(collection, query):
    if collection is None or not isinstance(collection, list):
        return None
    for item in collection:
        match = True
        for k, v in query.items():
            if item.get(k) != v:
                match = False
                break
        if match:
            return item
    return None

def safe_insert(collection, data):
    if collection is not None and isinstance(collection, list):
        if "_id" not in data:
            data["_id"] = str(uuid.uuid4())
        collection.append(data)
    else:
        print("Skipping DB insert")

def safe_update_one(collection, query, update, **kwargs):
    if collection is not None and isinstance(collection, list):
        item = safe_find_one(collection, query)
        if item is None and kwargs.get("upsert"):
            item = dict(query)
            collection.append(item)
        if item:
            if "$set" in update:
                item.update(update["$set"])
            if "$unset" in update:
                for k in update["$unset"]:
                    item.pop(k, None)
            if "$inc" in update:
                for k, v in update["$inc"].items():
                    item[k] = item.get(k, 0) + v
        return item
    return None

def safe_delete_many(collection, query):
    if collection is not None and isinstance(collection, list):
        to_remove = [item for item in collection if all(item.get(k) == v for k, v in query.items())]
        for item in to_remove:
            collection.remove(item)
        return len(to_remove)
    return None

def safe_find(collection, *args, **kwargs):
    if collection is not None and isinstance(collection, list):
        query = args[0] if args else {}
        return [item for item in collection if all(item.get(k) == v for k, v in query.items())]
    return []

def is_logged_in():
    return session.get("user_id") is not None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_user_id():
    return str(uuid.uuid4())


def utc_to_ist(dt_obj):
    if not dt_obj:
        return "-"
    if isinstance(dt_obj, str):
        return dt_obj
    try:
        # If naive, assume UTC
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=pytz.utc)
        ist_dt = dt_obj.astimezone(IST)
        return ist_dt.strftime("%d-%m-%Y %I:%M %p")
    except Exception:
        return str(dt_obj)


def generate_google_password(full_name: str) -> str:
    clean_name = re.sub(r"[^A-Za-z]", "", full_name)
    if not clean_name:
        clean_name = "User"
    random_digits = random.randint(10000, 99999)
    return f"{clean_name}@{random_digits}"


def refine_domain_label(raw_domain: str, resume_text: str = "", skills=None) -> str:
    """
    Convert broad domain labels (e.g., Engineering) into role-level domains
    using full resume text and extracted skills.
    """
    if not raw_domain:
        raw_domain = "Unknown"

    skills = skills or []
    text = f"{resume_text} {' '.join(skills)}".lower()
    base = raw_domain.strip()

    if base.lower() != "engineering":
        return base

    role_keywords = [
        (
            "Cyber Security",
            [
                "cyber security", "cybersecurity", "soc", "siem", "splunk",
                "vulnerability", "penetration testing", "pentest", "owasp",
                "incident response", "network security", "nmap", "burp suite",
            ],
        ),
        (
            "Software Engineer",
            [
                "software engineer", "software developer", "backend", "frontend",
                "full stack", "api", "microservices", "django", "flask",
                "spring boot", "react", "node", "typescript", "java",
                "python", "c++", "git", "rest",
            ],
        ),
        (
            "Data Scientist",
            [
                "data scientist", "machine learning", "deep learning", "nlp",
                "xgboost", "pytorch", "tensorflow", "scikit", "pandas",
                "model training", "feature engineering",
            ],
        ),
        (
            "DevOps Engineer",
            [
                "devops", "kubernetes", "docker", "jenkins", "ci/cd", "terraform",
                "ansible", "aws", "azure", "gcp", "monitoring", "prometheus",
            ],
        ),
        (
            "QA Engineer",
            [
                "qa engineer", "quality assurance", "test automation", "selenium",
                "cypress", "postman", "jmeter", "unit testing", "integration testing",
            ],
        ),
    ]

    best_label = "Engineering"
    best_score = 0
    for label, keywords in role_keywords:
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def get_boost_keywords_for_domain(predicted_domain: str, missing_skills=None):
    """
    Recommend high-impact keywords that generally improve ATS/domain match.
    Missing skills are prioritized first.
    """
    missing_skills = missing_skills or []
    domain = (predicted_domain or "").strip().lower()

    keyword_bank = {
        "cyber security": [
            "SIEM", "SOC", "Incident Response", "Vulnerability Assessment",
            "OWASP", "Penetration Testing", "Nmap", "Burp Suite",
            "Threat Modeling", "IAM",
        ],
        "software engineer": [
            "Data Structures", "System Design", "REST APIs", "Microservices",
            "Git", "Unit Testing", "CI/CD", "Docker", "SQL", "Cloud",
        ],
        "data scientist": [
            "Machine Learning", "Feature Engineering", "Model Evaluation",
            "Python", "Pandas", "Scikit-learn", "TensorFlow", "SQL",
            "A/B Testing", "Statistics",
        ],
        "devops engineer": [
            "Kubernetes", "Docker", "Terraform", "CI/CD", "Jenkins",
            "Monitoring", "Prometheus", "AWS", "Linux", "Ansible",
        ],
        "qa engineer": [
            "Test Automation", "Selenium", "Cypress", "API Testing",
            "Postman", "Regression Testing", "Test Cases", "CI/CD",
            "JIRA", "Performance Testing",
        ],
    }

    defaults = [
        "Quantified Impact", "Cross-functional Collaboration", "Ownership",
        "Problem Solving", "Communication",
    ]

    recommended = keyword_bank.get(domain, defaults)
    merged = []
    for kw in (missing_skills[:8] + recommended):
        if isinstance(kw, str) and kw and kw not in merged:
            merged.append(kw)
    return merged[:12]


def create_initial_stats(user_id: str):
    if stats_collection is None:
        return
    existing = safe_find_one(stats_collection, {"user_id": user_id})
    if not existing:
        safe_insert(stats_collection, {
            "user_id": user_id,
            "total_resumes": 0,
            "parsed_success": 0,
            "avg_match_score": 0,
            "processing_time": 0,
            "updated_at": datetime.now()
        })


def get_user_stats(user_id: str):
    if stats_collection is None:
        return {
            "total_resumes": 0,
            "parsed_success": 0,
            "avg_match_score": 0,
            "processing_time": 0
        }
    stats = safe_find_one(stats_collection, {"user_id": user_id})
    if not stats:
        create_initial_stats(user_id)
        stats = safe_find_one(stats_collection, {"user_id": user_id})
    
    # Final safety check
    if not stats:
        return {
            "total_resumes": 0,
            "parsed_success": 0,
            "avg_match_score": 0,
            "processing_time": 0
        }
    return stats


EMAIL_TYPES = {
    "OTP_VERIFY": "/send-otp",
    "RESET_PASSWORD": "/reset-password",
    "WELCOME": "/welcome",
}


def generate_otp():
    return str(random.randint(100000, 999999))


def send_mail(
    email: str,
    mail_type: str,
    otp: str = None,
    redirect: str = None,
):
    base_url = os.getenv("SENDMAIL_API_URL")

    if not base_url:
        return {"success": False, "message": "Mail service not configured"}

    payload = {
        "email": email,
        "type": mail_type
    }

    if otp:
        payload["otp"] = otp
    if redirect:
        payload["redirect"] = redirect

    try:
        response = requests.post(base_url, json=payload, timeout=20)

        if response.ok:
            return {"success": True, "message": "Mail sent"}

        print("MAIL URL:", base_url)
        print("MAIL PAYLOAD:", payload)
        print("MAIL STATUS:", response.status_code)
        print("MAIL RESPONSE:", response.text)
        
        return {
            "success": False,
            "message": response.json().get("message", "Mail failed")
        }

    except requests.RequestException:
        return {"success": False, "message": "Mail service unreachable"}


# ----------------------------
# Google reCAPTCHA verification
# ----------------------------

def verify_recaptcha(token):
    secret = os.getenv("RECAPTCHA_SECRET_KEY")
    if not secret:
        return False

    response = requests.post(
        "https://www.google.com/recaptcha/api/siteverify",
        data={
            "secret": secret,
            "response": token
        },
        timeout=10
    )

    result = response.json()
    return result.get("success", False)

# ----------------------------
# Genrate Hash
# ----------------------------

def generate_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


# ----------------------------
# File Integrity (MongoDB)
# ----------------------------

def generate_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


# ----------------------------
# DEV BYPASS (local testing only)
# ----------------------------
@app.route("/dev-login")
def dev_login():
    session.permanent = True
    session["user_id"] = "dev_user_001"
    session["user_email"] = "dev@test.local"
    session["name"] = "Dev Tester"
    session["profile_img"] = None
    session["show_personalization"] = False
    return redirect(url_for("dashboard"))


# ----------------------------
# Signup
# ----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        return render_template("signup.html")


    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if not name or not email or not password:
        flash("All fields are required!", "error")
        return redirect(url_for("signup"))

    name = name.strip()
    email = email.strip().lower()
    password = password.strip()

    if user_collection is None:
        flash("Database unavailable", "error")
        return redirect(url_for("signup"))

    existing = safe_find_one(user_collection, {"email": email})
    if existing:
        flash("Email already exists. Please sign in.", "error")
        return redirect(url_for("signin"))

    user_id = generate_user_id()
    hashed_pw = generate_password_hash(password)


    safe_insert(user_collection, {
        "user_id": user_id,
        "name": name,
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.now(),
        "last_updated": datetime.now(),
        "login_type": "manual",
        "profile_img": None,
        "is_verified": False,
        "personalized": False,
        "role": None,
        "subrole": None
    })

    create_initial_stats(user_id)

    # ----- OTP -----
    otp = generate_otp()
    otp_hash = generate_password_hash(otp)
    otp_expiry = datetime.now() + timedelta(minutes=10)

    safe_update_one(user_collection, 
        {"user_id": user_id},
        {"$set": {
            "otp_hash": otp_hash,
            "otp_expiry": otp_expiry
        }}
    )

    # ----- SEND MAIL -----
    mail_result = send_mail(
        email=email,
        mail_type="OTP_VERIFY",
        otp=otp
    )

    if not mail_result["success"]:
        
        flash("OTP send failed. Please try again.", "error" )
        return redirect(url_for("signup"))

    # ----- SESSION -----
    session["verify_email"] = email
    session["verify_allowed"] = True
    session.permanent = True

    flash("OTP sent to your email. Please verify your account.", "success")
    return redirect(url_for("verify_account_page"))

# ----------------------------
# Signin
# ----------------------------
@app.route("/signin", methods=["GET", "POST"])
def signin():
    # If already logged in → dashboard
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        if user_collection is None:
            flash("Database unavailable", "error")
            return redirect(url_for("signin"))

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        recaptcha_token = request.form.get("g-recaptcha-response")

        if not recaptcha_token or not verify_recaptcha(recaptcha_token):
            flash("reCAPTCHA verification failed. Please try again.", "error")
            return redirect(url_for("signin"))

        user = safe_find_one(user_collection, {"email": email})

        if not user:
            flash("Incorrect email id or password", "error")
            return redirect(url_for("signin"))
        
        if user.get("is_verified") is False:
            session["verify_allowed"] = True
            session["verify_email"] = email
            session.pop("otp_sent", None)  # IMPORTANT
            return redirect(url_for("verify_account_page"))

        if not user.get("password") or not check_password_hash(user["password"], password):
            flash("Incorrect email id or password", "error")
            return redirect(url_for("signin"))

        session.permanent = True
        session["user_id"] = user["user_id"]
        session["user_email"] = user["email"]
        session["name"] = user["name"]
        session["profile_img"] = user.get("profile_img")
        if not user.get("personalized"):
            session["show_personalization"] = True
        else:
            session["show_personalization"] = False
            
        return redirect(url_for("dashboard"))

    return render_template(
        "signin.html",
        recaptcha_site_key=os.getenv("RECAPTCHA_SITE_KEY")
    )


# ----------------------------
# Google Login
# ----------------------------
@app.route("/google-login")
def google_login():
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    google.authorize_access_token()
    
    resp = google.get("https://openidconnect.googleapis.com/v1/userinfo")
    userinfo = resp.json()
    
    email = userinfo.get("email")
    name = userinfo.get("name", "Google User")
    profile_img = userinfo.get("picture")

    if not email:
        flash("Google login failed!", "error")
        return redirect(url_for("signin"))

    existing = safe_find_one(user_collection, {"email": email})
             
    # FIXED BLOCK (indentation + logic)
    if not existing:
        raw_password = generate_google_password(name)
        hashed_password = generate_password_hash(raw_password)
        user_id = generate_user_id()

        safe_insert(user_collection, {
            "user_id": user_id,
            "name": name,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "login_type": "google",
            "profile_img": profile_img,
            "is_verified": True,
            "personalized": False,
            "role": None,
            "subrole": None
        })

        create_initial_stats(user_id)
        user = safe_find_one(user_collection, {"user_id": user_id})

        if not user.get("personalized"):
            session["show_personalization"] = True
        else:
            session["show_personalization"] = False
            
        flash(f"Google account created Password: {raw_password}", "success")

    else:
        user_id = existing.get("user_id")

        if not user_id:
            user_id = generate_user_id()
            safe_update_one(user_collection, 
                {"email": email},
                {"$set": {"user_id": user_id}}
            )
            create_initial_stats(user_id)

        safe_update_one(user_collection, 
            {"email": email},
            {"$set": {"profile_img": profile_img, "name": name, "last_updated": datetime.now(), "is_verified": True}}
        )
        
    session.permanent = True
    session["user_id"] = user_id
    session["user_email"] = email
    session["name"] = name
    session["profile_img"] = profile_img

    flash("Logged in with Google ", "success")
    
    return redirect(url_for("dashboard"))


# ----------------------------
# Verify
# ----------------------------

@app.route("/verifyaccount", methods=["GET"])
def verify_account_page():
    if not session.get("verify_allowed") or not session.get("verify_email"):
        flash("Unauthorized access.", "error")
        return redirect(url_for("signin"))

    email = session["verify_email"]

    # ✅ Send OTP only once (first visit)
    if not session.get("otp_sent"):
        run_sendotp_async(email)
        session["otp_sent"] = True
        flash("OTP sent to your email.", "success")

    return render_template(
        "verify_account.html",
        email=email,
        otp_sent=session["otp_sent"]
    )

@app.route("/verifyaccount", methods=["POST"])
def verify_account():
    if not session.get("verify_allowed") or not session.get("verify_email"):
        flash("Session expired. Please login again.", "error")
        return redirect(url_for("signin"))

    otp = request.form.get("otp", "").strip()
    email = session["verify_email"]

    if not otp.isdigit() or len(otp) != 6:
        flash("Invalid OTP format.", "error")
        return redirect(url_for("verify_account_page"))

    user = safe_find_one(user_collection, {"email": email})

    if not user or not user.get("otp_hash"):
        flash("OTP not found. Please resend.", "error")
        return redirect(url_for("verify_account_page"))

    if user.get("otp_expiry") < datetime.now():
        flash("OTP expired. Please resend.", "error")
        return redirect(url_for("verify_account_page"))

    if not check_password_hash(user["otp_hash"], otp):
        flash("Incorrect OTP.", "error")
        return redirect(url_for("verify_account_page"))

    safe_update_one(user_collection, 
        {"email": email},
        {
            "$set": {
                "is_verified": True,
                "last_updated": datetime.now()
            },
            "$unset": {
                "otp_hash": "",
                "otp_expiry": ""
            }                                       
        }
    )

    session.pop("verify_allowed", None)
    session.pop("verify_email", None)
    session.pop("otp_sent", None)

    flash("Email verified successfully. Please login.", "success")
    return redirect(url_for("signin"))



def run_sendotp_async(email):
    thread = threading.Thread(target=sendotp, args=(email,), daemon=True)
    thread.start()
    
    
def sendotp(email):
    otp = str(random.randint(100000, 999999))
    hashed_otp = generate_password_hash(otp)

    safe_update_one(user_collection, 
        {"email": email},
        {
            "$set": {
                "otp_hash": hashed_otp,
                "otp_expiry": datetime.now() + timedelta(minutes=10),
                "last_updated": datetime.now()
            }
        }
    )

    send_mail(
        email=email,
        mail_type="OTP_VERIFY",
        otp=otp
    )
    
# ----------------------------
# Resend OTP
# ----------------------------

@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    email = session.get("verify_email")

    if not email:
        return {"success": False, "message": "Session expired. Please login again."}, 401

    run_sendotp_async(email)

    return {
        "success": True,
        "message": "OTP resent successfully."
    }

# ----------------------------
# Forgot Password
# ----------------------------

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "GET":
        return render_template("forgot_password.html")
    
    email = request.form.get("email", "").strip().lower()

    if not email:
        flash("Email is required.", "error")
        return redirect(url_for("forgot_password"))
    
    user = safe_find_one(user_collection, {"email": email})

    # Prevent email enumeration
    if not user:
        flash("If this email exists, a reset link has been sent.", "success")
        return redirect(url_for("forgot_password"))

    now = datetime.now()

    # 🔒 Check if user is blocked
    block_until = user.get("reset_block_until")
    if block_until and block_until > now:
        remaining = int((block_until - now).total_seconds() / 60)
        flash(f"Too many reset attempts. Try again after {remaining} minutes.", "error")
        return redirect(url_for("forgot_password"))

    # Get current attempts
    attempts = user.get("reset_attempts", 0)

    if attempts >= 4:
        block_time = now + timedelta(minutes=20)

        safe_update_one(user_collection, 
            {"email": email},
            {"$set": {
                "reset_block_until": block_time,
                "reset_attempts": 0
            }}
        )

        flash("Too many reset attempts. Please try again after 20 minutes.", "error")
        return redirect(url_for("forgot_password"))

    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    safe_delete_many(reset_tokens_collection, {"user_id": user["user_id"]})

    safe_insert(reset_tokens_collection, {
        "user_id": user["user_id"],
        "token_hash": token_hash,
        "expires_at": now + timedelta(minutes=15),
        "used": False,
        "created_at": now
    })

    safe_update_one(user_collection, 
    {"user_id": user["user_id"]},
    {"$set": {
        "reset_attempts": 0,
        "reset_block_until": None
    }})

    reset_link = url_for("reset_password", token=token, _external=True)

    send_mail(
        email=email,
        mail_type="RESET_PASSWORD",
        redirect=reset_link
    )

    flash("If this email exists, a reset link has been sent.", "success")
    return redirect(url_for("forgot_password"))



@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token):

    token_hash = hashlib.sha256(token.encode()).hexdigest()

    reset_entry = safe_find_one(reset_tokens_collection, {
        "token_hash": token_hash,
        "used": False
    })

    if not reset_entry:
        flash("Invalid or expired reset link.", "error")
        return redirect(url_for("signin"))

    if reset_entry["expires_at"] < datetime.utcnow():
        flash("Reset link expired.", "error")
        return redirect(url_for("signin"))

    user = safe_find_one(user_collection, {"user_id": reset_entry["user_id"]})
    
    if not user:
        flash("Invalid reset link.", "error")
        return redirect(url_for("signin"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not (
            6 <= len(password) <= 15
            and re.search(r"[A-Z]", password)
            and re.search(r"[0-9]", password)
            and re.search(r"[@.&\-_]", password)
            and password == confirm
        ):
            flash("Password requirements not satisfied.", "error")
            return redirect(url_for("reset_password", token=token))

        hashed = generate_password_hash(password)

        safe_update_one(user_collection, 
            {"_id": user["_id"]},
            {"$set": {
                "password": hashed,
                "last_updated": datetime.utcnow()
            }}
        )

        # Mark token as used (single-use protection)
        safe_update_one(reset_tokens_collection, 
            {"_id": reset_entry["_id"]},
            {"$set": {"used": True}}
        )

        flash("Password reset successful. Please login.", "success")
        return redirect(url_for("signin"))

    return render_template("reset_password.html")



@app.route("/save-user-role", methods=["POST"])
def save_user_role():

    if not is_logged_in():
        return jsonify({"status": "error"}), 401

    data = request.get_json()

    role = data.get("role")
    subrole = data.get("subrole")

    user_id = session["user_id"]

    safe_update_one(user_collection, 
        {"user_id": user_id},
        {
            "$set": {
                "role": role,
                "subrole": subrole,
                "personalized": True,
                "last_updated": datetime.now()
            }
        }
    )

    session["show_personalization"] = False

    return jsonify({"status": "success"})


@app.route("/skip-personalization", methods=["POST"])
def skip_personalization():

    if not is_logged_in():
        return jsonify({"status": "error"}), 401

    session["show_personalization"] = False

    return jsonify({"status": "success"})

# ----------------------------
# Dashboard
# ----------------------------
@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]

    stats = get_user_stats(user_id)

    all_activities_raw = safe_find(activity_collection, {"user_id": user_id})
    # Sort by upload_date descending and take top 8
    all_activities_raw.sort(key=lambda x: x.get("upload_date", datetime.min), reverse=True)
    activities = all_activities_raw[:8]

    for act in activities:
        act["_id"] = str(act["_id"])
        # Prepare formatted date for display
        act["upload_date_local"] = utc_to_ist(act.get("upload_date"))
        # Prepare ISO string for JSON serialization (JS Chart)
        if isinstance(act.get("upload_date"), datetime):
            act["upload_date"] = act["upload_date"].isoformat()
        elif act.get("upload_date") is None:
            act["upload_date"] = None
        else:
            act["upload_date"] = str(act["upload_date"])

    parsed_percent = 0
    if stats.get("total_resumes", 0) > 0:
        parsed_percent = round((stats.get("parsed_success", 0) / stats["total_resumes"]) * 100)

    # Keep latest analysis visible across dashboard refreshes.
    last_result = session.get("last_result")

    # Prepare data for charts
    import collections
    parsed_over_time = collections.defaultdict(int)
    skill_freq = collections.defaultdict(int)
    
    all_activities = safe_find(activity_collection, {"user_id": user_id})
    for a in all_activities:
        # Over time - group by YYYY-MM-DD
        dt = a.get("upload_date")
        if dt:
            date_str = dt.strftime("%Y-%m-%d")
            parsed_over_time[date_str] += 1

        # Missing Skills frequency
        m_skills = a.get("missing_skills", [])
        if isinstance(m_skills, list):
            for skill in m_skills:
                if isinstance(skill, str) and len(skill) > 1:
                    skill_freq[skill] += 1

    # Sort dates chronologically
    sorted_dates = sorted(parsed_over_time.keys())
    chart_labels = sorted_dates[-7:] # Last 7 active days
    chart_values = [parsed_over_time[d] for d in chart_labels]
    
    # Top 5 missing skills
    top_skills = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    skill_labels = [s[0][:15] for s in top_skills]
    skill_values = [s[1] for s in top_skills]

    return render_template(
        "dashboard.html",
        page="dashboard",
        user_name=session.get("name", "User"),
        stats=stats,
        parsed_percent=parsed_percent,
        activities=activities,
        last_result=last_result,
        show_personalization=session.get("show_personalization", False),
        chart_labels=chart_labels,
        chart_values=chart_values,
        skill_labels=skill_labels,
        skill_values=skill_values
    )


# ----------------------------
# Upload page
# ----------------------------
@app.route("/upload")
def upload():
    if not is_logged_in():
        return redirect(url_for("signin"))
    return render_template("upload.html", page="upload")


@app.route("/store-analysis-result", methods=["POST"])
def store_analysis_result():
    if not is_logged_in():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    user_id = session["user_id"]
    payload = request.get_json(silent=True) or {}

    raw_domain = payload.get("predicted_domain", "Unknown")
    confidence_raw = payload.get("confidence", 0)
    all_probabilities = payload.get("all_probabilities", {})
    missing_keywords = payload.get("missing_keywords", [])
    suggestions = payload.get("suggestions", [])
    top_keywords = payload.get("top_keywords", [])
    final_score = payload.get("final_score")

    try:
        confidence = round(float(confidence_raw) * 100, 1)
    except (TypeError, ValueError):
        confidence = 0.0

    if final_score is None:
        final_score = round(confidence)

    if not isinstance(missing_keywords, list):
        missing_keywords = []
    if not isinstance(suggestions, list):
        suggestions = []
    if not isinstance(top_keywords, list):
        top_keywords = []
    if not isinstance(all_probabilities, dict):
        all_probabilities = {}

    feedback = "\n".join(suggestions)
    predicted_domain = refine_domain_label(
        raw_domain=raw_domain,
        resume_text=payload.get("full_resume_text", ""),
        skills=top_keywords,
    )
    boost_keywords = get_boost_keywords_for_domain(predicted_domain, missing_keywords)

    # Save for dashboard card (trim to avoid cookie overflow).
    session["last_result"] = {
        "name": session.get("name", "User"),
        "predicted_domain": predicted_domain,
        "confidence": confidence,
        "final_score": final_score,
        "feedback": feedback[:500],
        "missing_skills": missing_keywords[:10],
        "missing_keywords": missing_keywords[:15],
        "top_keywords": ", ".join(top_keywords[:8]),
        "strengths": [],
        "suggestions": suggestions[:8],
        "boost_keywords": boost_keywords,
        "all_probabilities": all_probabilities,
        "latency_ms": payload.get("latency_ms", 0),
        "type": "resume",
    }

    # Add activity for dashboard table/charts.
    if activity_collection is not None:
        safe_insert(
            activity_collection,
            {
                "user_id": user_id,
                "candidate_name": session.get("name", "User"),
                "name": session.get("name", "User"),
                "email": session.get("user_email", ""),
                "domain": predicted_domain,
                "score": final_score,
                "feedback": feedback,
                "missing_skills": missing_keywords[:10],
                "upload_date": datetime.now(),
                "status": "Success",
                "match_score": final_score,
                "file_name": payload.get("file_name", "Client Upload"),
            },
        )

    process_time = 0
    try:
        latency_ms = float(payload.get("latency_ms", 0) or 0)
        process_time = round(latency_ms / 1000, 2)
    except (TypeError, ValueError):
        process_time = 0

    if stats_collection is not None:
        safe_update_one(
            stats_collection,
            {"user_id": user_id},
            {
                "$inc": {"total_resumes": 1, "parsed_success": 1},
                "$set": {"processing_time": process_time, "updated_at": datetime.now()},
            },
            upsert=True,
        )

    return jsonify({"status": "ok"})


# ----------------------------
# Upload Resume and ParserAI
# ----------------------------
@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]

    if "resume" not in request.files:
        flash("No file selected!", "error")
        return redirect(url_for("upload"))

    file = request.files["resume"]
    if file.filename == "":
        flash("No file selected!", "error")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash("Only PDF, DOC, DOCX, PNG, JPG files are allowed!", "error")
        return redirect(url_for("upload"))

    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)

    if file_length > MAX_FILE_SIZE:
        flash("File too large! Max file size is 28MB.", "error")
        return redirect(url_for("upload"))

    job_description = request.form.get("job_description", "")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    
    # ----------------------------
    # Generate & Store Hash in MongoDB
    # ----------------------------
    file_hash = generate_hash(filepath)

    if file_integrity_collection is not None:
        safe_insert(file_integrity_collection, {
            "user_id": user_id,
            "filename": filename,
            "filehash": file_hash,
            "uploaded_at": datetime.now()
        })
    else:
        print("Skipping DB insert (MongoDB not connected)")


    parser_url = os.getenv("PARSERAI_URL", "https://dyno0126-cv-mind-analyzer.hf.space/upload-analyze")

    user_name = session.get("name", "User")

    status = "Pending"
    match_score = 0
    predicted_domain = ""
    feedback = ""
    missing_skills = []
    candidate_email = ""
    candidate_name = user_name

    start_time = time.time()
    try:
        with open(filepath, 'rb') as f:
            # Field name must match FastAPI param: `file: UploadFile = File(...)`
            files = {'file': (filename, f, file.mimetype)}
            
            response = requests.post(
                parser_url,
                files=files,
                timeout=120
            )

        process_time = round(time.time() - start_time, 2)

        if response.status_code == 200:
            result = response.json()
            status = "Success"

            raw_domain = result.get("predicted_domain", "Unknown")
            confidence = result.get("confidence", 0)
            
            strengths = result.get("skills_found", [])
            missing_skills = result.get("missing_skills", [])
            extracted_text = result.get("full_resume_text", "")
            
            suggestions = result.get("suggestions", [])
            feedback = "\n".join(suggestions)
            
            keywords_list = result.get("keywords", [])
            keyword_str = ", ".join(keywords_list[:5])
            
            match_score = result.get("final_score", 0)
            predicted_domain = refine_domain_label(
                raw_domain=raw_domain,
                resume_text=extracted_text,
                skills=strengths,
            )
            boost_keywords = get_boost_keywords_for_domain(predicted_domain, missing_skills)

            candidate_name = user_name
            candidate_email = session.get("user_email", "")

            # Store full result in session to show on dashboard (Truncate arrays to prevent 4KB Cookie Overflow)
            session['last_result'] = {
                "name": candidate_name,
                "predicted_domain": predicted_domain,
                "confidence": round(confidence * 100, 1),
                "final_score": match_score,
                "feedback": feedback[:500] if feedback else "",  # Cap feedback string length
                "missing_skills": missing_skills[:10],
                "top_keywords": keyword_str,
                "strengths": strengths[:10],
                "boost_keywords": boost_keywords,
                "latency_ms": result.get("latency_ms", 0),
            }

        else:
            status = "Error"
            match_score = None

    except requests.exceptions.Timeout:
        process_time = round(time.time() - start_time, 2)
        status = "Pending"
        match_score = None
    except Exception:
        process_time = round(time.time() - start_time, 2)
        status = "Error"
        match_score = None

    if activity_collection is not None:
        safe_insert(activity_collection, {
            "user_id": user_id,
            "candidate_name": candidate_name,
            "name": candidate_name,
            "email": candidate_email,
            "domain": predicted_domain,
            "score": match_score,
            "feedback": feedback,
            "missing_skills": missing_skills,
            "upload_date": datetime.now(),
            "status": status,
            "match_score": match_score,
            "file_name": filename
        })
    else:
        print("Skipping DB insert (MongoDB not connected)")

    if stats_collection is not None:
        safe_update_one(stats_collection, 
            {"user_id": user_id},
            {
                "$inc": {"total_resumes": 1},
                "$set": {"processing_time": process_time, "updated_at": datetime.now()}
            },
            upsert=True
        )

        if status == "Success":
            safe_update_one(stats_collection, 
                {"user_id": user_id},
                {"$inc": {"parsed_success": 1}},
                upsert=True
            )

    if activity_collection is not None and stats_collection is not None:
        # Simplified query for in-memory list (safe_find doesn't support $ne)
        all_user_activities = safe_find(activity_collection, {"user_id": user_id})
        scores = [a["match_score"] for a in all_user_activities if a.get("match_score") is not None]

        avg = round(sum(scores) / len(scores), 2) if scores else 0

        safe_update_one(stats_collection, 
            {"user_id": user_id},
            {"$set": {"avg_match_score": avg}}
        )

    flash("Resume uploaded processing...", "success")
    return redirect(url_for("dashboard"))


# ----------------------------
# LinkedIn Profile Analysis
# ----------------------------
@app.route("/analyze-linkedin", methods=["POST"])
def analyze_linkedin():
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]
    linkedin_text = request.form.get("linkedin_text", "").strip()
    job_description = request.form.get("job_description", "").strip()

    if not linkedin_text or len(linkedin_text) < 30:
        flash("Please paste more LinkedIn profile content!", "error")
        return redirect(url_for("upload"))

    # Use the same CV Mind Analyzer HF Space — /analyze-text accepts JSON {"text": "..."}
    parser_url = os.getenv("PARSERAI_URL", "https://dyno0126-cv-mind-analyzer.hf.space/upload-analyze")
    analyze_url = parser_url.replace("/upload-analyze", "/analyze-text")

    start_time = time.time()
    try:
        payload = {"text": linkedin_text}
        response = requests.post(analyze_url, json=payload, timeout=120)
        process_time = round(time.time() - start_time, 2)

        if response.status_code == 200:
            result = response.json()
            
            # Use 'predicted_domain' and 'confidence' per the FastAPI schema
            predicted_domain = result.get("predicted_domain", "Unknown")
            confidence = result.get("confidence", 0)
            
            # Extract top 2 for display in keywords logic if needed or fallback
            all_probs = result.get("all_probabilities", {})
            sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)[:2]
            top_skills_display = [f"{k} ({v*100:.1f}%)" for k, v in sorted_probs]
            
            # Store in session for immediate display
            session['last_result'] = {
                "name": result.get("name", "Candidate"),
                "predicted_domain": predicted_domain,
                "confidence": round(confidence * 100, 1),
                "final_score": round(confidence * 100), # Using confidence as proxy for match score
                "skills": result.get("skills", []), # Fallbacks
                "experience": [],
                "education": [],
                "top_keywords": ", ".join(top_skills_display),
                "latency_ms": process_time * 1000,
                "backend": result.get("backend", "docker_fastapi"),
                "type": "linkedin"  # Mark as LinkedIn result
            }

            # Update stats
            if stats_collection is not None:
                safe_update_one(stats_collection, 
                    {"user_id": user_id},
                    {
                        "$inc": {"total_resumes": 1, "parsed_success": 1},
                        "$set": {"processing_time": process_time, "updated_at": datetime.now()}
                    },
                    upsert=True
                )
            
            # Add to activity
            if activity_collection is not None:
                safe_insert(activity_collection, {
                    "user_id": user_id,
                    "candidate_name": result.get("name", "Unknown"),
                    "domain": result.get("predicted_domain", "Unknown"),
                    "score": result.get("match_score", 0),
                    "upload_date": datetime.now(),
                    "status": "Success",
                    "type": "linkedin",
                    "file_name": "LinkedIn Parse"
                })

            flash("LinkedIn profile analyzed successfully!", "success")
        else:
            flash(f"LinkedIn analysis failed: {response.text}", "error")

    except Exception as e:
        flash(f"Error connecting to analyzer: {str(e)}", "error")

    return redirect(url_for("dashboard"))



# ----------------------------
# Verify File Integrity (Mongo)
# ----------------------------
@app.route("/verify/<filename>")
def verify_file(filename):
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(filepath):
        return "File missing from server"

    record = safe_find_one(file_integrity_collection, {
        "user_id": user_id,
        "filename": filename
    })

    if not record:
        return "File not found in integrity database"

    stored_hash = record.get("filehash")
    current_hash = generate_hash(filepath)

    if current_hash == stored_hash:
        return "✅ Integrity Verified: File is Original"
    else:
        return "🚨 Integrity Violated: File has been Modified"


# ----------------------------
# Other pages
# ----------------------------
@app.route("/parsed")
def parsed():
    if not is_logged_in():
        return redirect(url_for("signin"))
    return render_template("parsed.html", page="parsed")


@app.route("/analytics")
def analytics():
    if not is_logged_in():
        return redirect(url_for("signin"))
    return render_template("analytics.html", page="analytics")


@app.route("/settings")
def settings():
    if not is_logged_in():
        return redirect(url_for("signin"))
    return render_template("settings.html", page="settings")


# ----------------------------
# Logout
# ----------------------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


# ----------------------------
# No back after google login 
# ----------------------------
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    app.run(debug=True)
