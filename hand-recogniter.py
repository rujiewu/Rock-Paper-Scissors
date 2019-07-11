import hand_classifier
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

im_width, im_height = (cap.get(3), cap.get(4))
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret, frame = cap.read()
    if ret:
        image, result = hand_classifier.detect(frame)
        print(result)
        cv2.imshow('RPS', image)
        cv2.moveWindow('RPS', 0, 0)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break




