from ultralytics import YOLO
import cv2
import cvzone

model_path = "best.pt"
detector = YOLO(model_path)

product_labels = ['Ariel', 'Coca Cola', 'Colgate', 'Fanta', 'Kurkure',
                  'Lays Masala', 'Lays Mexican', 'Lifebuoy Soap',
                  'Sunsilk Shampoo', 'Vaseline Lotion']

detected_counts = {label: 0 for label in product_labels}
detection_threshold = 0.4


def analyze_frame(frame):
    global detected_counts
    detection_results = detector(frame, stream=True)
    detected_counts = {label: 0 for label in product_labels}

    for result in detection_results:
        bounding_boxes = result.boxes
        for box in bounding_boxes:
            coords = list(map(int, box.xyxy[0]))
            confidence_score = box.conf[0].item()
            class_index = int(box.cls[0])
            current_label = product_labels[class_index]

            if confidence_score > detection_threshold:
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 3)
                cvzone.putTextRect(frame, f'{current_label} {confidence_score:.2f}',
                                   (coords[0], coords[1] - 10), scale=3.0, thickness=4,
                                   colorB=(0, 255, 0), colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5)
                detected_counts[current_label] += 1

    text_offset = 50
    for label, count in detected_counts.items():
        if count > 0:
            cvzone.putTextRect(frame, f'{label}: {count}', (10, text_offset), scale=3.0, thickness=4,
                               colorB=(255, 255, 255), colorT=(0, 0, 0), colorR=(255, 0, 0), offset=5)
            text_offset += 50

    return frame


def main():
    global detected_counts
    detected_counts = {label: 0 for label in product_labels}
    user_choice = input("Press '1' for webcam input or '2' for image input: ")
    display_resolution = (1450, 800)

    if user_choice == '1':
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            print("Error: Unable to access the webcam.")
            return

        print("Press 'q' to exit.")
        while True:
            success, frame = webcam.read()
            if not success:
                print("Error: Unable to read from the webcam.")
                break

            frame = analyze_frame(frame)
            frame = cv2.resize(frame, display_resolution)
            cv2.imshow("Webcam View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()

    elif user_choice == '2':
        image_path = input("Provide the path to the image file: ")
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Unable to read the image file. Check the path.")
            return

        frame = analyze_frame(frame)
        frame = cv2.resize(frame, display_resolution)
        cv2.imshow("Image View", frame)
        print("Press any key to close the image.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
