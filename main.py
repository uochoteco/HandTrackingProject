import cv2
import mediapipe

mpHands = mediapipe.solutions.hands
hands = mediapipe.Hands()
mpDraw = mediapipe.solutions.drawing_utils

capture = cv2.VideoCapture(0)

while True:
    works, img = capture.read()
    if not works:
        print("Camera didn't work idiot")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = hands.process(img_rgb)

    if output.multi_hand_landmarks:
        for hand_lms in output.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_lms, mpHands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand TRacking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
        