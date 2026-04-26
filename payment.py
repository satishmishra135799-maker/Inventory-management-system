"""
payment.py — UPI QR code generation.

Generates a UPI deep-link QR code for the bill total.
Customer scans with any UPI app (GPay, PhonePe, Paytm, etc.)
"""

import qrcode
import io
import base64

# Store UPI config — change these to real store values
UPI_ID   = "store@upi"
PAYEE    = "Smart Grocery Store"


def build_upi_url(amount: float, txn_note: str = "Grocery Bill") -> str:
    """Build UPI payment deep-link string."""
    return (
        f"upi://pay?pa={UPI_ID}"
        f"&pn={PAYEE.replace(' ', '%20')}"
        f"&am={amount:.2f}"
        f"&cu=INR"
        f"&tn={txn_note.replace(' ', '%20')}"
    )


def generate_qr_png(amount: float) -> bytes:
    """Return UPI QR code as PNG bytes."""
    url = build_upi_url(amount)
    qr  = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def generate_qr_base64(amount: float) -> str:
    """Return UPI QR code as base64 string (for embedding in HTML/PDF)."""
    return base64.b64encode(generate_qr_png(amount)).decode()
