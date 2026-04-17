import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)
detector = vision.HandLandmarker.create_from_options(options)

capture = cv2.VideoCapture(0)

while capture.isOpened():
    works, image = capture.read()
    if not works:
        break
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mpImage = mp.Image(image_format = mp.ImageFormat.SRGB, data = image_rgb)
    detectorOutput = detector.detect(mpImage)
    cv2.imshow('Hand Tracking - Tasks API', cv2.flip(image, 1))
    
    if cv2.waitKey(1) & 0&FF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()