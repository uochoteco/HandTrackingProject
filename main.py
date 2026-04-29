import math
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)

def countFingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    fingersUp = 0

    for i in range(4):
        if hand_landmarks[tips[i]].y < hand_landmarks[knuckles[i]].y:
           fingersUp = fingersUp + 1

    if abs(hand_landmarks[4].x - hand_landmarks[2].x) > 0.05:
        fingersUp = fingersUp + 1

    return fingersUp

def checkCircle(points):
    if len(points) < 20:
        return False

    start = points[0]
    end = points[-1]
    distance = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    
    if distance > 50:
        return False
    
    perimeter = 0
    for i in range (1, len(points)):
        perimeter = perimeter + math.sqrt((points[i][0] - points[i - 1][0])**2 + (points[i][1] - points[i - 1][1])**2)

    xCord = [i[0] for i in points]
    yCord = [i[1] for i in points]
    width = max(xCord) - min(xCord)
    height = max(yCord) - min(yCord)

    ratio = min(width, height)/max(width, height)
    if ratio > 0.8:
        return False
    else:
        return True
    
def isFist(hand_landmarks):
    if ((hand_landmarks[8].y > hand_landmarks[6].y) and (hand_landmarks[12].y > hand_landmarks[10].y) and (hand_landmarks[16].y > hand_landmarks[14].y) and (hand_landmarks[20].y > hand_landmarks[18].y)):
        return True
    else:
        return False

with vision.HandLandmarker.create_from_options(options) as detector:

    capture = cv2.VideoCapture(0)
    timers = [0.0, 0.0]
    handsAssigned =[False, False]
    color = [255, 0, 0]
    drawingPoints = []
    timeIndexDown = 0
    shapeMaking = False
    sOrigin = None
    sDiam = 0
    sFinal = False

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
            
            presentHands = [h[0].category_name for h in output.handedness]

            for index, hand_in_frame in enumerate(output.hand_landmarks):
                label = presentHands[index]
                fingers = countFingers(hand_in_frame)

                if label == "Left":
                    if fingers == 1 and not handsAssigned[0]:
                        if timers[0] == 0:
                            timers[0] = time.time()
                        elif time.time() - timers[0] > 2:
                            print("hand one marked")
                            handsAssigned[0] = True
                        print(time.time() - timers[0])
                    elif not handsAssigned[0]:
                        timers[0] = 0
                    progress = 0
                    if handsAssigned[0]:
                        progress = 1
                    elif timers[0] > 0:
                        progress = min((time.time() - timers[0]) / 2, 1.0)
                    color = ((int)(255 * (1 - progress)),0 , (int)(255 * (progress)))
                    
                if label == "Right":
                    if fingers == 2 and not handsAssigned[1]:
                        if timers[1] == 0:
                            timers[1] = time.time()
                        elif time.time() - timers[1] > 2:
                            print("hand two marked")
                            handsAssigned[1] = True
                        print(time.time() - timers[1])
                    elif not handsAssigned[1]:
                        timers[1] = 0
                    progress = 0
                    if handsAssigned[1]:
                        progress = 1
                    elif timers[1] > 0:
                        progress = min((time.time() - timers[1]) / 2, 1.0)
                    color = ((int)(255 * (1 - progress)), (int)(255 * (progress)), 0)

                if label == "Left" and handsAssigned[0] and handsAssigned[1]:
                    indexTip = hand_in_frame[8]

                    if fingers == 1:
                        xPix = int(indexTip.x * image.shape[1])
                        yPix = int(indexTip.y * image.shape[0])
                        drawingPoints.append((xPix, yPix))
                        timeIndexDown = time.time()

                    elif time.time() - timeIndexDown > 0.5:
                        drawingPoints = []

                for connection in connections:
                    startID = connection[0]
                    endID = connection[1]
                    pOne = hand_in_frame[startID]
                    pTwo = hand_in_frame[endID]
                    vOne = (int(pOne.x * image.shape[1]), int(pOne.y * image.shape[0]))
                    vTwo = (int(pTwo.x * image.shape[1]), int(pTwo.y * image.shape[0]))
                    cv2.line(image, vOne, vTwo, (255, 255, 0), 2)
                
                for landmark in hand_in_frame:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, color, -1)
            
            if len(drawingPoints) > 2:
                drawingColor = (255, 255, 255)

            if checkCircle(drawingPoints):
                print("Circle Found!")
                drawingColor = (0, 255, 0)
                if shapeMaking:
                    for hIndex, hLabel in enumerate(presentHands):
                        if hLabel == "Right":
                            middle = output.hand_landmarks[hIndex][9]
                            sOrigin = (middle.x, middle.y, middle.z)
                            shapeMaking = True
                            print("origin found")

            if shapeMaking and not sFinal:
                for hIndex, hLabel in enumerate(presentHands):
                    if hLabel == "Right":
                        middle = output.hand_landmarks[hIndex][9]
                        diameter = math.sqrt((middle.x - sOrigin[0])**2 + (middle.y - sOrigin[1])**2, (middle.z - sOrigin[2])**2)
                        sDiam = (int)(diameter  * image.shape[1])
                        if isFist(output.hand_landmarks[hIndex]):
                            sFinal = True
                            shapeMaking = False
                    
            if sFinal:
                for hIndex, hLabel in enumerate(presentHands):
                    if hLabel == "Right":
                        middle = output.hand_landmarks[hIndex][9]
                        centerPx = ((sOrigin[0] + middle.x)/2)
                        centerPy = ((sOrigin[1] + middle.y)/2)
                        sRadius = sDiam/2
                        cv2.circle(image, (centerPx, centerPy), sRadius, (255, 0, 255), 2)
                        cv2.ellipse(image, (centerPx, centerPy), (sRadius, sRadius // 3) ,0 ,0 ,360, (255, 0, 255), 1)

            if len(drawingPoints) > 2:
                for i in range(1, len(drawingPoints)):
                        cv2.line(image, drawingPoints[i - 1], drawingPoints[i], (drawingColor[0], drawingColor[1], drawingColor[2]), 2)

        cv2.imshow('Hand Tracking - Tasks API', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

