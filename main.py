import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)
with vision.HandLandmarker.create_from_options(options) as detector:

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        works, image = capture.read()
        if not works:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format = mp.ImageFormat.SRGB, data = image_rgb)
        output = detector.detect(mpImage)

        if output.hand_landmarks:
            for hand_in_frame in output.hand_landmarks:
                for landmark in hand_in_frame:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                    
        cv2.imshow('Hand Tracking - Tasks API', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()