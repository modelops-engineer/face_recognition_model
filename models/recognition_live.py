# open camera
# detect faces
# generate embedding
# compare with stored embeddings
# show name if match
# show UNKNOWN otherwise


import os
import numpy as np
import pickle as pkl
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import cv2


cap = cv2.VideoCapture(0)

mtcnn = MTCNN(keep_all=True, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

database_path = os.path.join(os.getcwd(), "database\\embeddings.pkl")

try:
    with open(database_path, 'rb') as f:
        database = pkl.load(f)
except:
    print('could not open database')

if not cap.isOpened():
    print('Could not open camera')
    exit()
    

def recognize(embedding, threshold):
    best_match = None
    best_score = -1

    for names, embeddings in database.items():
        for stored_embedding in embeddings:
            similarity = cosine_similarity(embedding,stored_embedding)[0][0]
            if similarity > best_score:
                best_score = similarity
                best_match = names
       
    if best_score > threshold:
        return names, best_score
           
    return 'Unknown', best_score

while True:
    ret, frame = cap.read()

    if not ret:
        break
    display = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face_crop = rgb_frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_tensor = mtcnn(face_crop)

            if face_tensor is None:
                continue
            
            embedding = resnet(face_tensor).detach().numpy()

            name, score = recognize(embedding,0.6)

            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{name} ({score})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8, color, 2)
            
        cv2.imshow('face recognition', display)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            exit()

cap.release()
cv2.destroyAllWindows()
