import cv2
import time
from ultralytics import YOLO
import serial

model = YOLO("best.pt")

class_to_char = {
    "trash": b'T',
    "metal": b'M',
    "paper": b'P',
    "plastic": b'L',
    "glass": b'G',
    "clothes": b'C',
    "shoes": b'S',
    "battery": b'B',
    "cardboard": b'D',
    "biological": b'O'
}

# ----- SERIAL SETUP -----
# ser = serial.Serial('COM5', 9600)  # Uncomment and set correct COM port
# time.sleep(2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame)

        pred_class_id = results[0].probs.top1
        class_name = model.names[pred_class_id]
        conf = results[0].probs.top1conf.item()

        print(f"Detected: {class_name} ({conf:.2f})")

        display_text = f"{class_name} ({conf:.2f})"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Live Classification", frame)

        if class_name in class_to_char:
            print(f"Sent to Arduino: {class_to_char[class_name].decode()}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1) 

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()