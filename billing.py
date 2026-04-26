"""
billing.py — Bill computation + PDF invoice with UPI QR code.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from datetime import datetime
import io

PRICE_DB = {
    'Ariel':           299,
    'Coca Cola':        40,
    'Colgate':          89,
    'Fanta':            40,
    'Kurkure':          20,
    'Lays Masala':      20,
    'Lays Mexican':     20,
    'Lifebuoy Soap':    55,
    'Sunsilk Shampoo': 179,
    'Vaseline Lotion': 199,
}


def compute_bill(detected_counts: dict, price_db: dict = None) -> tuple[list, float]:
    db = price_db if price_db is not None else PRICE_DB
    items, total = [], 0.0
    for product, count in detected_counts.items():
        if count > 0 and product in db:
            unit  = db[product]
            sub   = unit * count
            items.append({'product': product, 'qty': count,
                          'unit_price': unit, 'subtotal': sub})
            total += sub
    return items, total


def generate_pdf_invoice(detected_counts: dict, output_path: str = None,
                         price_db: dict = None) -> str:
    from payment import generate_qr_png   # imported here to avoid circular issues

    items, total = compute_bill(detected_counts, price_db=price_db)
    if not items:
        print("Cart empty — invoice not generated.")
        return None

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"invoice_{ts}.pdf"

    doc      = SimpleDocTemplate(output_path, pagesize=A4,
                                  rightMargin=2*cm, leftMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
    styles   = getSampleStyleSheet()
    elements = []

    # Header
    elements.append(Paragraph("Smart Grocery Checkout", styles['Title']))
    elements.append(Paragraph("Automated Billing System · Patent Pending",
                               styles['Normal']))
    elements.append(Paragraph(
        f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        styles['Normal']))
    elements.append(Spacer(1, 0.4*cm))

    # Items table
    tdata = [['Product', 'Qty', 'Unit (₹)', 'Total (₹)']]
    for it in items:
        tdata.append([it['product'], str(it['qty']),
                      f"{it['unit_price']:.2f}", f"{it['subtotal']:.2f}"])
    tdata.append(['', '', 'TOTAL', f"₹{total:.2f}"])

    tbl = Table(tdata, colWidths=[7*cm, 2*cm, 3.5*cm, 3.5*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  colors.HexColor('#1a73e8')),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS',(0, 1), (-1, -2), [colors.whitesmoke, colors.white]),
        ('BACKGROUND',    (0,-1), (-1, -1), colors.HexColor('#e8f0fe')),
        ('FONTNAME',      (2,-1), (-1, -1), 'Helvetica-Bold'),
        ('GRID',          (0, 0), (-1, -1), 0.4, colors.grey),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.6*cm))

    # UPI QR code
    qr_bytes = generate_qr_png(total)
    qr_img   = RLImage(io.BytesIO(qr_bytes), width=4*cm, height=4*cm)
    elements.append(Paragraph("Scan to Pay via UPI", styles['Normal']))
    elements.append(Spacer(1, 0.2*cm))
    elements.append(qr_img)
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph("Thank you for shopping!", styles['Normal']))

    doc.build(elements)
    print(f"Invoice saved: {output_path}")
    return output_path


def print_bill_console(detected_counts: dict, price_db: dict = None):
    items, total = compute_bill(detected_counts, price_db=price_db)
    if not items:
        print("No products detected.")
        return
    print("\n" + "="*45)
    print(f"{'GROCERY BILL':^45}")
    print(f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S'):^45}")
    print("="*45)
    print(f"{'Product':<20} {'Qty':>4} {'Price':>8} {'Total':>8}")
    print("-"*45)
    for it in items:
        print(f"{it['product']:<20} {it['qty']:>4} "
              f"{it['unit_price']:>8.2f} {it['subtotal']:>8.2f}")
    print("="*45)
    print(f"{'TOTAL':>34} {total:>8.2f}")
    print("="*45 + "\n")
