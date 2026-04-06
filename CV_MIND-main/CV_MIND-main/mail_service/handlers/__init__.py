from .otp_mail import build_otp_email
from .reset_mail import build_reset_email

HANDLERS = {
    "OTP_VERIFY": build_otp_email,
    "RESET_PASSWORD": build_reset_email
}
