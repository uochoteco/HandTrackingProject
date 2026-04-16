import cv2
import mediapipe
from mediapipe.python.solutions import hands as mpHands
from mediapipe.python.solutions import drawing_utils as mpDraw

hands = mpHands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

capture = cv2.VideoCapture(0)

while True:
    works, img = capture.read()
    if not works:
        print("Camera didn't work idiot")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_lms, mpHands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand TRacking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
        