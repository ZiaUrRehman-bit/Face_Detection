import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fd

cTime = 0
pTime = 0

cam = cv2.VideoCapture(0)
detection = fd.faceDetect()

while True:
    Success, frame = cam.read()
    frame = detection.findFace(frame)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
