import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #newly added

model = load_model("model/mask_model.h5")

#to use a pretrained model for face detection
face_model = cv2.dnn.readNet(
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    "face_detector/deploy.prototxt"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    print(frame)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    face_model.setInput(blob)
    detections = face_model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            # face = face / 255.0
            face=preprocess_input(face) #newly added
            face = np.reshape(face, (1, 224, 224, 3))

            pred = model.predict(face)
            # label = "Mask" if pred[0][0] > pred[0][1] else "No Mask"
            # color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            mask_confidence = np.max(pred)

            if mask_confidence < 0.6:
                label = "Uncertain"
                color = (0, 255, 255)  # Yellow

            else:
                if pred[0][0] > pred[0][1]:
                    label = "Mask"
                    color = (0, 255, 0)
                else:
                    label = "No Mask"
                    color = (0, 0, 255)
                    
            text = f"{label} ({mask_confidence:.2f})"


            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
