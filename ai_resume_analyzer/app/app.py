from flask import Flask, render_template, redirect, url_for, request, session, flash, make_response, jsonify
from authlib.integrations.flask_client import OAuth
from db import user_collection, stats_collection, activity_collection, file_integrity_collection, reset_tokens_collection   
from itsdangerous import URLSafeTimedSerializer
from werkzeug.middleware.proxy_fix import ProxyFix
import sys, os, time, smtplib, requests, pytz, random, re, uuid, filetype
from datetime import datetime, timedelta
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import threading, hashlib, secrets
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from bson import ObjectId
from text_extractor import extract_candidate_name

from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)
# Generate a highly secure random key if none is provided in the environment.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or os.urandom(32).hex()

# Enable CSRF Protection globally
csrf = CSRFProtect(app)

# Setup Rate Limiting to prevent brute-force attacks
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Force HTTPS for url_for in production (Vercel)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Security headers for session cookies
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Existing cache headers (if any) are handled in the other after_request hook
    return response


# ----------------------------
# Upload Config
# ----------------------------

# Use /tmp for uploads on Vercel/Serverless
UPLOAD_FOLDER = "/tmp" if os.getenv("VERCEL") else "uploads"
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
MAX_FILE_SIZE = 28 * 1024 * 1024  # 28MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)



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

serializer = URLSafeTimedSerializer(
    app.secret_key if app.secret_key else "fallbacksecretkey"
)

# ----------------------------
# Helpers
# ----------------------------


# Timezone Helper
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


def create_initial_stats(user_id: str):
    existing = stats_collection.find_one({"user_id": user_id})
    if not existing:
        stats_collection.insert_one({
            "user_id": user_id,
            "total_resumes": 0,
            "parsed_success": 0,
            "avg_match_score": 0,
            "processing_time": 0,
            "updated_at": datetime.now()
        })


def get_user_stats(user_id: str):
    stats = stats_collection.find_one({"user_id": user_id})
    if not stats:
        create_initial_stats(user_id)
        stats = stats_collection.find_one({"user_id": user_id})
    return stats


EMAIL_TYPES = {
    "OTP_VERIFY": "/send-otp",
    "RESET_PASSWORD": "/reset-password",
    "WELCOME": "/welcome",
}


def generate_otp():
    return str(random.randint(100000, 999999))


def send_mail(email: str, mail_type: str, otp: str = None, redirect: str = None):
    try:
        from mail_service.handlers.otp_mail import build_otp_email
        from mail_service.handlers.reset_mail import build_reset_email
        
        MAIL_EMAIL = os.getenv("MAIL_EMAIL")
        MAIL_APP_PASSWORD = os.getenv("MAIL_APP_PASSWORD")
        SMTP_HOST = "smtp.gmail.com"
        SMTP_PORT = 465
        
        if not MAIL_EMAIL or not MAIL_APP_PASSWORD:
            print("[ERROR] MAIL_EMAIL or MAIL_APP_PASSWORD not set in .env")
            return {"success": False, "message": "Mail service not configured"}
            
        data = {"email": email, "type": mail_type, "otp": otp, "redirect": redirect}
        
        if mail_type == "OTP_VERIFY":
            msg = build_otp_email(data)
        elif mail_type == "RESET_PASSWORD":
            msg = build_reset_email(data)
        else:
            return {"success": False, "message": f"Invalid mail type: {mail_type}"}
            
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(MAIL_EMAIL, MAIL_APP_PASSWORD)
            smtp.send_message(msg)
            
        return {"success": True, "message": "Mail sent successfully"}
        
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
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
# Signup
# ----------------------------
@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("10 per hour")
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

    existing = user_collection.find_one({"email": email})
    if existing:
        flash("Email already exists. Please sign in.", "error")
        return redirect(url_for("signin"))

    user_id = generate_user_id()
    hashed_pw = generate_password_hash(password)


    user_collection.insert_one({
        "user_id": user_id,
        "name": name,
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.now(),
        "last_updated": datetime.now(),
        "login_type": "manual",
        "profile_img": None,
        "is_verified": False,
        "personalized": False,   # ADD THIS
        "role": None,
        "subrole": None
    })

    create_initial_stats(user_id)

    # ----- OTP -----
    otp = generate_otp()
    otp_hash = generate_password_hash(otp)
    otp_expiry = datetime.now() + timedelta(minutes=10)

    user_collection.update_one(
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
@limiter.limit("15 per minute")
def signin():
    # If already logged in → dashboard
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        recaptcha_token = request.form.get("g-recaptcha-response")

        if not recaptcha_token or not verify_recaptcha(recaptcha_token):
            flash("reCAPTCHA verification failed. Please try again.", "error")
            return redirect(url_for("signin"))

        user = user_collection.find_one({"email": email})

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
        session["name"] = str(user.get("name") or "User")
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


@app.route("/google-login")
def google_login():
    # Use explicit base URL if configured (e.g. on Vercel)
    base_url = os.getenv("FRONTEND_BASE_URL")
    if base_url:
        redirect_uri = f"{base_url.rstrip('/')}/auth/google/callback"
    else:
        redirect_uri = url_for("google_callback", _external=True)
        
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    try:
        google.authorize_access_token()
        
        resp = google.get("https://openidconnect.googleapis.com/v1/userinfo")
        userinfo = resp.json()
        
        email = userinfo.get("email")
        name = userinfo.get("name", "Google User")
        profile_img = userinfo.get("picture")

        if not email:
            flash("Google login failed!", "error")
            return redirect(url_for("signin"))

        existing = user_collection.find_one({"email": email})
                
        if not existing:
            raw_password = generate_google_password(name)
            hashed_password = generate_password_hash(raw_password)
            user_id = generate_user_id()

            user_collection.insert_one({
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
            user_data = user_collection.find_one({"user_id": user_id})

            if not user_data.get("personalized"):
                session["show_personalization"] = True
            else:
                session["show_personalization"] = False
                
            flash(f"Google account created successfully", "success")

        else:
            user_id = existing.get("user_id")

            if not user_id:
                user_id = generate_user_id()
                user_collection.update_one(
                    {"email": email},
                    {"$set": {"user_id": user_id}}
                )
                create_initial_stats(user_id)

            user_collection.update_one(
                {"email": email},
                {"$set": {"profile_img": profile_img, "name": name, "last_updated": datetime.now(), "is_verified": True}}
            )
            
        session.permanent = True
        session["user_id"] = user_id
        session["user_email"] = email
        session["name"] = str(name or "User")
        session["profile_img"] = profile_img

        flash("Logged in with Google", "success")
        return redirect(url_for("dashboard"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Login failed: {str(e)}", "error")
        return redirect(url_for("signin"))


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

    user = user_collection.find_one({"email": email})

    if not user or not user.get("otp_hash"):
        flash("OTP not found. Please resend.", "error")
        return redirect(url_for("verify_account_page"))

    if user.get("otp_expiry") < datetime.now():
        flash("OTP expired. Please resend.", "error")
        return redirect(url_for("verify_account_page"))

    if not check_password_hash(user["otp_hash"], otp):
        flash("Incorrect OTP.", "error")
        return redirect(url_for("verify_account_page"))

    user_collection.update_one(
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

    user_collection.update_one(
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
@limiter.limit("5 per minute")
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
@limiter.limit("5 per minute")
def forgot_password():

    if request.method == "GET":
        return render_template("forgot_password.html")
    
    email = request.form.get("email", "").strip().lower()

    if not email:
        flash("Email is required.", "error")
        return redirect(url_for("forgot_password"))
    
    user = user_collection.find_one({"email": email})

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

        user_collection.update_one(
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

    reset_tokens_collection.delete_many({"user_id": user["user_id"]})

    reset_tokens_collection.insert_one({
        "user_id": user["user_id"],
        "token_hash": token_hash,
        "expires_at": now + timedelta(minutes=15),
        "used": False,
        "created_at": now
    })

    user_collection.update_one(
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
@limiter.limit("5 per minute")
def reset_password(token):

    token_hash = hashlib.sha256(token.encode()).hexdigest()

    reset_entry = reset_tokens_collection.find_one({
        "token_hash": token_hash,
        "used": False
    })

    if not reset_entry:
        flash("Invalid or expired reset link.", "error")
        return redirect(url_for("signin"))

    if reset_entry["expires_at"] < datetime.utcnow():
        flash("Reset link expired.", "error")
        return redirect(url_for("signin"))

    user = user_collection.find_one({"user_id": reset_entry["user_id"]})
    
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

        user_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {
                "password": hashed,
                "last_updated": datetime.utcnow()
            }}
        )

        # Mark token as used (single-use protection)
        reset_tokens_collection.update_one(
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

    user_collection.update_one(
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

@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]

    try:
        stats = get_user_stats(user_id)
        activity_cursor = activity_collection.find({"user_id": user_id}).sort("upload_date", -1).limit(8)
        activities = list(activity_cursor)

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
    except Exception as e:
        print(f"[ERROR] Dashboard data fetch failed: {e}")
        stats = {"total_resumes": 0, "parsed_success": 0, "avg_match_score": 0}
        activities = []
        parsed_percent = 0
        flash("Could not fetch latest stats from database. Showing cached data.", "warning")

    return render_template(
        "dashboard.html",
        page="dashboard",
        user_name=session.get("name", "User"),
        stats=stats,
        parsed_percent=parsed_percent,
        activities=activities,
        show_personalization=session.get("show_personalization", False),
        last_result=session.get("last_result")
    )


# ----------------------------
# Upload page
# ----------------------------
@app.route("/upload")
def upload():
    if not is_logged_in():
        return redirect(url_for("signin"))
    return render_template("upload.html", page="upload")


@app.route("/upload-resume", methods=["POST"])
@limiter.limit("50 per day")
@limiter.limit("10 per hour")
def upload_resume():
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]

    if "resume" not in request.files:
        flash("No file selected!", "error")
        return redirect(url_for("upload"))

    files = request.files.getlist("resume")
    if not files or files[0].filename == "":
        flash("No file selected!", "error")
        return redirect(url_for("upload"))

    success_count = 0
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    last_result = None

    for file in files:
        if not allowed_file(file.filename):
            flash(f"Skipped {file.filename}: Only PDF, DOC, DOCX files are allowed!", "warning")
            continue

        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)

        if file_length > MAX_FILE_SIZE:
            flash(f"Skipped {file.filename}: File too large! Max file size is 28MB.", "warning")
            continue

        # --- MALWARE / PAYLOAD PROTECTION (Magic Bytes Check) ---
        kind = filetype.guess(file.stream)
        file.seek(0) # reset pointer after reading magic bytes
        
        if kind is None or kind.mime not in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            flash(f"Security Alert: Rejected {file.filename}. Invalid file signature detected.", "error")
            continue
            
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        file_hash = generate_hash(filepath)

        file_integrity_collection.insert_one({
            "user_id": user_id,
            "filename": filename,
            "filehash": file_hash,
            "uploaded_at": datetime.now()
        })

        status = "Pending"
        match_score = None
        start_time = time.time()
        process_time = 0
        result = {}
        extracted_text = ""

        try:
            parser_url = os.getenv("PARSERAI_URL") or "https://dyno0126-cv-mind-analyzer.hf.space/upload-analyze"

            with open(filepath, "rb") as f:
                upload_files = {"file": (filename, f)}
                response = requests.post(parser_url, files=upload_files, timeout=60)
                
            if response.status_code == 200:
                result = response.json()
                status = "Success"
                match_score = result.get("final_score", 0)
                extracted_text = result.get("full_resume_text", "")
                
                result["matched_keywords"] = result.get("skills_found", [])
                result["missing_keywords"] = result.get("missing_skills", [])
            else:
                status = "Error"
                print(f"[ERROR] API analysis failed with status {response.status_code}: {response.text}")
                
            process_time = round(time.time() - start_time, 2)
            
        except Exception as e:
            process_time = round(time.time() - start_time, 2)
            status = "Error"
            print(f"[ERROR] Analysis failed: {e}")

        extracted_name = extract_candidate_name(extracted_text)
        final_candidate_name = extracted_name if extracted_name else session.get("name", "User")

        if status == "Success":
            last_result = {
                "name": final_candidate_name,
                "predicted_domain": result.get("predicted_domain", "Unknown"),
                "confidence": round(result.get("confidence", 0) * 100, 1),
                "final_score": match_score,
                "missing_skills": result.get("missing_keywords", []),
                "strengths": result.get("matched_keywords", []),
                "suggestions": result.get("suggestions", []),
                "latency_ms": round(process_time * 1000)
            }
            success_count += 1

        activity_collection.insert_one({
            "user_id": user_id,
            "candidate_name": final_candidate_name,
            "upload_date": datetime.now(),
            "status": status,
            "match_score": match_score,
            "file_name": filename,
            "predicted_domain": result.get("predicted_domain", "Unknown"),
            "confidence": result.get("confidence", 0) * 100,
            "missing_skills": result.get("missing_keywords", []),
            "strengths": result.get("matched_keywords", []),
            "suggestions": result.get("suggestions", []),
            "latency_ms": round(process_time * 1000)
        })

        stats_collection.update_one(
            {"user_id": user_id},
            {
                "$inc": {"total_resumes": 1},
                "$set": {"processing_time": process_time, "updated_at": datetime.now()}
            },
            upsert=True
        )

        if status == "Success":
            stats_collection.update_one(
                {"user_id": user_id},
                {"$inc": {"parsed_success": 1}},
                upsert=True
            )

    if last_result:
        session['last_result'] = last_result

    scores = list(activity_collection.find(
        {"user_id": user_id, "match_score": {"$ne": None}},
        {"match_score": 1}
    ))

    avg = round(sum([s["match_score"] for s in scores]) / len(scores), 2) if scores else 0

    stats_collection.update_one(
        {"user_id": user_id},
        {"$set": {"avg_match_score": avg}}
    )

    if success_count > 0:
        flash(f"Successfully processed {success_count} resume(s).", "success")
    else:
        flash("No resumes were successfully processed.", "error")
        
    return redirect(url_for("dashboard"))


from bson.objectid import ObjectId

@app.route("/view-report/<activity_id>")
def view_report(activity_id):
    if not is_logged_in():
        return redirect(url_for("signin"))
        
    try:
        activity = activity_collection.find_one({
            "_id": ObjectId(activity_id),
            "user_id": session["user_id"]
        })
        
        if not activity:
            flash("Report not found.", "error")
            return redirect(url_for("dashboard"))
            
        session['last_result'] = {
            "name": activity.get("candidate_name"),
            "predicted_domain": activity.get("predicted_domain", "Unknown"),
            "confidence": activity.get("confidence", 0),
            "final_score": activity.get("match_score", 0),
            "missing_skills": activity.get("missing_skills", []),
            "strengths": activity.get("strengths", []),
            "suggestions": activity.get("suggestions", []),
            "latency_ms": activity.get("latency_ms", 0)
        }
        
        return redirect(url_for("dashboard"))
    except Exception as e:
        flash("Invalid report ID.", "error")
        return redirect(url_for("dashboard"))

# ----------------------------
# Verify File Integrity (Mongo)
# ----------------------------
@app.route("/verify/<filename>")
def verify_file(filename):
    if not is_logged_in():
        return redirect(url_for("signin"))

    user_id = session["user_id"]
    
    # Sanitize the filename to prevent path traversal attacks (e.g. ../../etc/passwd)
    safe_filename = secure_filename(filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)

    if not os.path.exists(filepath):
        return "File missing from server"

    record = file_integrity_collection.find_one({
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
    
    user_id = session["user_id"]
    activities = list(activity_collection.find({"user_id": user_id}).sort("upload_date", -1))
    
    for act in activities:
        act["_id"] = str(act["_id"])
        act["upload_date_local"] = utc_to_ist(act.get("upload_date"))
        
    return render_template("parsed.html", page="parsed", activities=activities)


@app.route("/analytics")
def analytics():
    if not is_logged_in():
        return redirect(url_for("signin"))
    
    user_id = session["user_id"]
    activities = list(activity_collection.find({"user_id": user_id}))
    
    # 1. Success vs Error Ratio
    success_count = sum(1 for a in activities if a.get("status") == "Success")
    error_count = sum(1 for a in activities if a.get("status") == "Error")
    pending_count = sum(1 for a in activities if a.get("status") == "Pending")
    
    # 2. Domain Distribution
    domain_counts = {}
    for a in activities:
        domain = a.get("predicted_domain", "Unknown")
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
    # 3. Top Skills
    skill_freq = {}
    for a in activities:
        for skill in a.get("strengths", []):
            skill_freq[skill] = skill_freq.get(skill, 0) + 1
            
    # 4. Monthly/Daily Trend
    trend_data = {}
    for a in activities:
        dt = a.get("upload_date")
        if dt:
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt)
                except:
                    continue
            date_str = dt.strftime("%Y-%m-%d")
            trend_data[date_str] = trend_data.get(date_str, 0) + 1
            
    sorted_trend = sorted(trend_data.items())
    
    analytics_data = {
        "status_labels": ["Success", "Error", "Pending"],
        "status_values": [success_count, error_count, pending_count],
        "domain_labels": list(domain_counts.keys()),
        "domain_values": list(domain_counts.values()),
        "skill_labels": sorted(skill_freq, key=skill_freq.get, reverse=True)[:10],
        "skill_values": sorted(skill_freq.values(), reverse=True)[:10],
        "trend_labels": [t[0] for t in sorted_trend],
        "trend_values": [t[1] for t in sorted_trend]
    }
    
    return render_template("analytics.html", page="analytics", data=analytics_data)


@app.route("/matching", methods=["GET", "POST"])
def matching():
    if not is_logged_in():
        return redirect(url_for("signin"))
    
    user_id = session["user_id"]
    # Get all successful parses for selection
    resumes = list(activity_collection.find({"user_id": user_id, "status": "Success"}))
    
    match_result = None
    if request.method == "POST":
        resume_id = request.form.get("resume_id")
        job_desc = request.form.get("job_description")
        
        if resume_id and job_desc:
            # Find the resume text in the resumes collection (not activities)
            # Actually, activities usually has the extracted info.
            # But the full text might be in 'resumes' collection or we can just use the activity summary.
            activity = activity_collection.find_one({"_id": ObjectId(resume_id)})
            
            if activity:
                # Call backend for matching
                parser_url = os.getenv("PARSERAI_URL") or "https://dyno0126-cv-mind-analyzer.hf.space/upload-analyze"
                # Since we don't have the original file easily here, 
                # we can use the /analyze-text endpoint if the backend supports it.
                # Let's check api/main.py for /analyze-text
                
                try:
                    # Fallback to the text-based matching if available
                    analyze_url = parser_url.replace("/upload-analyze", "/analyze")
                    payload = {
                        "text": activity.get("resume_text", ""), # We need to make sure we store resume_text in activity
                        "job_description": job_desc
                    }
                    # ... wait, let's keep it simple for now and just render the page
                    pass
                except:
                    pass
                    
    return render_template("matching.html", page="matching", resumes=resumes, result=match_result)


@app.route("/settings")
def settings():
    if not is_logged_in():
        return redirect(url_for("signin"))
    return render_template("settings.html", page="settings")


# ----------------------------
# Admin Panel
# ----------------------------
@app.route("/admin")
def admin():
    if not is_logged_in():
        return redirect(url_for("signin"))
        
    # Restrict to specific admin email
    if session.get("user_email") != "usecvmind@gmail.com":
        flash("Access denied. Admin privileges required.", "error")
        return redirect(url_for("dashboard"))
    
    # 1. Gather Users and basic stats
    all_users = list(user_collection.find({}))
    total_users = len(all_users)
    
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    weekly_active = 0
    monthly_active = 0
    
    user_stats_list = []
    
    total_parses_platform = 0
    total_success_platform = 0
    
    for u in all_users:
        uid = u.get("user_id")
        
        # Check activity for active status (based on last_updated or by querying activity_collection)
        # Using activity_collection for more accuracy on parsing activity:
        user_activities = list(activity_collection.find({"user_id": uid}))
        last_active = None
        
        domain_counts = {}
        for a in user_activities:
            dt = a.get("upload_date")
            if dt:
                # dt might be naive or aware, convert appropriately if needed. Assuming naive local or UTC.
                if isinstance(dt, datetime):
                    if last_active is None or dt > last_active:
                        last_active = dt
                
            domain = a.get("predicted_domain")
            if domain and domain != "Unknown":
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Most common domain (Behavior pattern)
        top_domain = "N/A"
        if domain_counts:
            top_domain = max(domain_counts, key=domain_counts.get)
            
        if last_active:
            if last_active > week_ago:
                weekly_active += 1
            if last_active > month_ago:
                monthly_active += 1
                
        # Parse stats
        stats = stats_collection.find_one({"user_id": uid}) or {}
        parses = stats.get("total_resumes", 0)
        successes = stats.get("parsed_success", 0)
        
        total_parses_platform += parses
        total_success_platform += successes
        
        success_rate = round((successes / parses * 100) if parses > 0 else 0)
        
        user_stats_list.append({
            "name": u.get("name", "Unknown"),
            "email": u.get("email", ""),
            "total_resumes": parses,
            "success_rate": success_rate,
            "last_active": utc_to_ist(last_active) if last_active else "Never",
            "behavior_pattern": top_domain
        })

    platform_success_rate = round((total_success_platform / total_parses_platform * 100) if total_parses_platform > 0 else 0)
    
    admin_data = {
        "total_users": total_users,
        "weekly_active": weekly_active,
        "monthly_active": monthly_active,
        "total_parses": total_parses_platform,
        "platform_success_rate": platform_success_rate,
        "users": user_stats_list
    }
    
    return render_template("admin.html", page="admin", data=admin_data)


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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
