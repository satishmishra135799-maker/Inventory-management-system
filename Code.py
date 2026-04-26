"""
Code.py — Automated Grocery Detection & Billing System
Uses YOLOv8L for detection + DeepSORT for tracking.
Each physical product is counted exactly once (not per-frame).

Controls (webcam mode):
  q — quit
  b — generate bill (console + PDF)
  r — reset cart / new customer
"""

from ultralytics import YOLO
import cv2
import cvzone
from tracker import ProductTracker
from billing import generate_pdf_invoice, print_bill_console

MODEL_PATH  = "best (1).pt"
CONF_THRESH = 0.4

PRODUCT_LABELS = [
    'Ariel', 'Coca Cola', 'Colgate', 'Fanta', 'Kurkure',
    'Lays Masala', 'Lays Mexican', 'Lifebuoy Soap',
    'Sunsilk Shampoo', 'Vaseline Lotion'
]

detector = YOLO(MODEL_PATH)
tracker  = ProductTracker()


def run_detections(frame) -> list[dict]:
    """Run YOLOv8 on a frame, return filtered detection dicts."""
    results = detector(frame, stream=True, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = PRODUCT_LABELS[int(box.cls[0])]
            detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'label': label})
    return detections


def draw_overlay(frame, detections: list[dict], cart: dict[str, int]):
    """Draw bounding boxes and cart summary onto frame."""
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cvzone.putTextRect(frame, f"{d['label']} {d['conf']:.2f}",
                           (x1, y1 - 10), scale=2.0, thickness=3,
                           colorB=(0, 255, 0), colorT=(255, 255, 255),
                           colorR=(0, 255, 0), offset=5)

    # Cart summary (top-left)
    y_off = 40
    cvzone.putTextRect(frame, "CART", (10, y_off), scale=2.0, thickness=3,
                       colorR=(0, 0, 200), offset=4)
    y_off += 45
    for label, count in cart.items():
        cvzone.putTextRect(frame, f"{label}: {count}", (10, y_off),
                           scale=1.8, thickness=2,
                           colorB=(255, 255, 255), colorT=(0, 0, 0),
                           colorR=(200, 200, 200), offset=4)
        y_off += 40

    # Controls hint (bottom)
    h = frame.shape[0]
    cv2.putText(frame, "B=Bill  R=Reset  Q=Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame


def webcam_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    tracker.reset()
    print("Webcam started. B=Bill  R=Reset  Q=Quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = run_detections(frame)
        cart = tracker.update(detections, frame)
        frame = draw_overlay(frame, detections, cart)
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Grocery Checkout", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            if cart:
                print_bill_console(cart)
                generate_pdf_invoice(cart)
            else:
                print("Cart is empty.")
        elif key == ord('r'):
            tracker.reset()
            print("Cart reset — ready for next customer.")

    cap.release()
    cv2.destroyAllWindows()


def image_mode():
    path = input("Image path: ").strip()
    frame = cv2.imread(path)
    if frame is None:
        print("Cannot read image.")
        return

    tracker.reset()
    detections = run_detections(frame)
    # For a static image, run update once (no temporal tracking needed)
    # Use raw per-class counts directly
    cart = {}
    for d in detections:
        cart[d['label']] = cart.get(d['label'], 0) + 1

    frame = draw_overlay(frame, detections, cart)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Grocery Checkout", frame)

    print_bill_console(cart)
    generate_pdf_invoice(cart)

    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("1 = Webcam  |  2 = Image : ").strip()
    if choice == '1':
        webcam_mode()
    elif choice == '2':
        image_mode()
    else:
        print("Invalid choice.")
