import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)

def countFingers(hand_landmarks):
    print("inCountFunction")
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    fingersUp = 0

    for i in range(4):
        if hand_landmarks[tips[i]].y < hand_landmarks[knuckles[i]].y:
           fingersUp = fingersUp + 1

    if abs(hand_landmarks[4].x - hand_landmarks[2].x) > 0.05:
        fingersUp = fingersUp + 1

    return fingersUp

with vision.HandLandmarker.create_from_options(options) as detector:

    capture = cv2.VideoCapture(0)
    handOneTimer = 0
    handTwoTimer = 0
    assignedHands = {}

    while capture.isOpened():
        works, image = capture.read()
        if not works:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format = mp.ImageFormat.SRGB, data = image_rgb)
        output = detector.detect(mpImage)

        if output.hand_landmarks:
            for hand_in_frame in output.hand_landmarks:
                connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
                    (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
                
                for connection in connections:
                    startID = connection[0]
                    endID = connection[1]
                    pOne = hand_in_frame[startID]
                    pTwo = hand_in_frame[endID]
                    vOne = (int(pOne.x * image.shape[1]), int(pOne.y * image.shape[0]))
                    vTwo = (int(pTwo.x * image.shape[1]), int(pTwo.y * image.shape[0]))
                    cv2.line(image, vOne, vTwo, (255, 0, 0), 2)
                
                for landmark in hand_in_frame:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            for index, hand_in_frame in enumerate(output.hand_landmarks):
                fingers = countFingers(hand_in_frame)

                if fingers == 2:
                    if index == 0:
                        print(time.time() - handOneTimer)
                        if handOneTimer == 0:
                            handOneTimer = time.time()
                        elif time.time() - handOneTimer > 2:
                            print("hand one marked")

                    if index == 1:
                        print(time.time() - handOneTimer)
                        if handTwoTimer == 0:
                            handTwoTimer = time.time()
                        elif time.time() - handTwoTimer > 2:
                            print("hand two marked")
                else:
                    if index == 0:
                        handOneTimer = 0
                    if index == 1:
                        handTwoTimer = 0
                        
        cv2.imshow('Hand Tracking - Tasks API', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

