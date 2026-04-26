"""
app.py — Flask web server for Automated Grocery Checkout System.

Checkout routes:
  GET  /              → checkout UI
  GET  /video_feed    → MJPEG camera stream
  GET  /cart          → cart JSON
  POST /remove        → remove one item
  POST /checkout      → finalize, save sale, generate PDF
  GET  /qr            → UPI QR PNG for current total
  GET  /download_invoice
  POST /reset

Dashboard routes:
  GET  /dashboard             → owner dashboard UI
  GET  /dashboard/products    → product list JSON
  POST /dashboard/products    → add/update product
  POST /dashboard/products/delete → delete product
  GET  /dashboard/sales       → sales summary JSON
  GET  /dashboard/revenue     → daily revenue JSON
"""

import sys, types
if 'pkg_resources' not in sys.modules:
    _pkg = types.ModuleType('pkg_resources')
    _pkg.require = lambda *a, **k: None
    sys.modules['pkg_resources'] = _pkg

import cv2
import threading
from flask import Flask, Response, jsonify, request, render_template, send_file
from ultralytics import YOLO
from tracker import ProductTracker
from billing import generate_pdf_invoice, print_bill_console
from db import get_price_db, record_sale, get_products, upsert_product, \
               delete_product, sales_summary, daily_revenue, get_variants

app = Flask(__name__)

MODEL_PATH  = "best (1).pt"
CONF_THRESH = 0.4
PRODUCT_LABELS = [
    'Ariel', 'Coca Cola', 'Colgate', 'Fanta', 'Kurkure',
    'Lays Masala', 'Lays Mexican', 'Lifebuoy Soap',
    'Sunsilk Shampoo', 'Vaseline Lotion'
]

detector = YOLO(MODEL_PATH)
tracker  = ProductTracker()
lock     = threading.Lock()
last_pdf = None
variant_prices: dict[str, float] = {}  # SKU label → price override


def detect(frame):
    results = detector(frame, stream=True, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = PRODUCT_LABELS[int(box.cls[0])]
            dets.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'label': label})
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    return frame, dets


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, dets = detect(frame)
        with lock:
            tracker.update(dets, frame)
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    cap.release()


# ── Checkout ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cart')
def cart():
    price_db = get_price_db()
    with lock:
        c = dict(tracker.cart)
    def price_of(k):
        return variant_prices.get(k, price_db.get(k, 0))
    total = sum(price_of(k) * v for k, v in c.items())
    items = [{'name': k, 'qty': v, 'price': price_of(k),
              'subtotal': price_of(k) * v} for k, v in c.items()]
    return jsonify({'items': items, 'total': total})


@app.route('/remove', methods=['POST'])
def remove():
    with lock:
        ok = tracker.remove_item(request.json.get('label'))
    return jsonify({'success': ok})


@app.route('/checkout', methods=['POST'])
def checkout():
    global last_pdf
    price_db = get_price_db()
    # Merge variant overrides into price_db for this transaction
    merged_prices = {**price_db, **variant_prices}
    with lock:
        cart_snap = dict(tracker.cart)
    if not cart_snap:
        return jsonify({'error': 'Cart is empty'}), 400
    print_bill_console(cart_snap, price_db=merged_prices)
    path = generate_pdf_invoice(cart_snap, price_db=merged_prices)
    last_pdf = path
    record_sale(cart_snap, merged_prices)
    with lock:
        tracker.reset()
    variant_prices.clear()
    return jsonify({'success': True, 'pdf': '/download_invoice'})


@app.route('/download_invoice')
def download_invoice():
    if last_pdf:
        return send_file(last_pdf, as_attachment=True)
    return jsonify({'error': 'No invoice'}), 404


@app.route('/qr')
def qr():
    from payment import generate_qr_png
    price_db = get_price_db()
    with lock:
        c = dict(tracker.cart)
    total = sum(price_db.get(k, 0) * v for k, v in c.items())
    if total <= 0:
        return jsonify({'error': 'Cart empty'}), 400
    return Response(generate_qr_png(total), mimetype='image/png')


@app.route('/variants')
def variants():
    """Return size variants for a brand."""
    brand = request.args.get('brand', '')
    return jsonify(get_variants(brand))


@app.route('/pending')
def pending():
    """Return the next brand awaiting variant selection (if any)."""
    with lock:
        brand = tracker.pending[0] if tracker.pending else None
    return jsonify({'brand': brand})


@app.route('/dismiss_pending', methods=['POST'])
def dismiss_pending():
    """Remove the first item from the pending queue."""
    with lock:
        if tracker.pending:
            tracker.pending.pop(0)
    return jsonify({'success': True})


@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    """Manually add a specific SKU variant to cart."""
    d     = request.json
    label = d.get('label')   # e.g. "Colgate 100g"
    price = float(d.get('price', 0))
    with lock:
        tracker.cart[label] = tracker.cart.get(label, 0) + 1
        if tracker.pending:
            tracker.pending.pop(0)   # dismiss the brand that triggered this popup
    variant_prices[label] = price
    return jsonify({'success': True})


@app.route('/reset', methods=['POST'])
def reset():
    with lock:
        tracker.reset()
    return jsonify({'success': True})


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/dashboard/products', methods=['GET'])
def dash_products():
    return jsonify(get_products())


@app.route('/dashboard/products', methods=['POST'])
def dash_upsert_product():
    d = request.json
    upsert_product(d['name'], float(d['price']), int(d['stock']))
    return jsonify({'success': True})


@app.route('/dashboard/products/delete', methods=['POST'])
def dash_delete_product():
    delete_product(request.json['name'])
    return jsonify({'success': True})


@app.route('/dashboard/sales')
def dash_sales():
    days = int(request.args.get('days', 7))
    return jsonify(sales_summary(days))


@app.route('/dashboard/revenue')
def dash_revenue():
    days = int(request.args.get('days', 7))
    return jsonify(daily_revenue(days))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
