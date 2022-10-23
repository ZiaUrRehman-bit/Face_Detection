import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:

    Success, frame = cam.read()
    imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imRGB)
    # print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(frame, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)

            cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            cv2.putText(frame, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

cam.release()
cv2.destroyAllWindows()

print("Code Completed!")