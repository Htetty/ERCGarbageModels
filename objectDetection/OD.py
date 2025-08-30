import cv2
from ultralytics import YOLO
# import serial, time 

model = YOLO("yolo11s.pt")
coco_to_group = {
    "book": "paper",

    "scissors": "metal", "fork": "metal", "knife": "metal", "spoon": "metal",
    "remote": "metal", "cell phone": "metal", "laptop": "metal", "keyboard": "metal",
    "microwave": "metal", "refrigerator": "metal", "oven": "metal", "toaster": "metal",

    "bottle": "trash", "cup": "trash",
    "banana": "trash", "apple": "trash", "orange": "trash", "broccoli": "trash",
    "carrot": "trash", "sandwich": "trash", "hot dog": "trash", "pizza": "trash",
    "donut": "trash", "cake": "trash", "toothbrush": "trash", "mouse": "trash", "tv": "trash",
    "handbag": "trash", "backpack": "trash", "suitcase": "trash", "tie": "trash", "umbrella": "trash",
}

class_to_char = {
    "trash": b'T',
    "metal": b'M',
    "paper": b'P',
}

CONF_THRESH = 0.35

# ----- SERIAL SETUP (optional) -----
# ser = serial.Serial('COM5', 9600)
# time.sleep(2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=CONF_THRESH)
        boxes = results[0].boxes

        chosen_group = None
        best_conf = -1.0

        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            name = model.names[cls_id]
            group = coco_to_group.get(name)
            if group and conf > best_conf:
                best_conf = conf
                chosen_group = group

        if chosen_group:
            text = f"{chosen_group} ({best_conf:.2f})"
            color = (0, 255, 0) if chosen_group != "trash" else (0, 200, 255)
            cv2.putText(frame, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "no relevant object", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        annotated = results[0].plot()

        if chosen_group:
            cv2.putText(annotated, f"[GROUP] {chosen_group}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("COCO detection → grouped classifier", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()