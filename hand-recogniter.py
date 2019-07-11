import hand_classifier
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

im_width, im_height = (cap.get(3), cap.get(4))
num_hands_detect = 1
font = cv2.FONT_HERSHEY_COMPLEX
flag = False

while True:
    try:
        ret, image_np = cap.read()
        if ret:
            retgesture = model.guess_gesture(model.image_preprocess(image_np))
        flag = True
    except Exception as err:
        print("Did not detect hand, put hand within the camera's frame!")
        continue

        print(model.gesture_postprocess(retgesture))
        cv2.imshow('RPS', image_np)
        cv2.moveWindow('RPS', 0, 0)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

