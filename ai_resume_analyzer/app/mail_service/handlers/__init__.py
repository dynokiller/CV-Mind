from .otp_mail import build_otp_email
from .reset_mail import build_reset_email
from .announcement_mail import build_announcement_email

HANDLERS = {
    "OTP_VERIFY": build_otp_email,
    "RESET_PASSWORD": build_reset_email,
    "ANNOUNCEMENT": build_announcement_email
}
