import cv2
from facenet_pytorch import MTCNN

cap = cv2.VideoCapture(0)
mtcnn = MTCNN(keep_all=True, device='cpu')


# if camera does not open
if not cap.isOpened():
    print('Camera not accessible')
    exit()

while True:
    ret, frame = cap.read()
    # ret : bool value if the frame is available
    # frame : an image array vector captured based on default frames per second
    frame = cv2.flip(frame,1)
    if not ret:
        break
    
    # convert BRG to RGB
    # OpenCV works in BRG but deep learning models work in RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)    

    cv2.imshow('frame', frame)

    # to exit press q
    if cv2.waitKey(1) & 0XFF == ord('q'):
        exit()

cap.release()
cv2.destroyAllWindows()