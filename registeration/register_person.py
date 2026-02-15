# open embeddings
# ask name
# create array to store the embeddings with a name
# open camera
# detect face
# press C
# quality checks
# generate embedding
# store
# repeat

import os
import sys
import cv2
import numpy as np
import pickle as pkl
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

path = os.path.join(os.getcwd(),'database\\embeddings.pkl')


cap = cv2.VideoCapture(0)
mtcnn = MTCNN(keep_all=False, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

try:
    with open(path, 'rb') as f:
        database = pkl.load(f)
        print('opened the database')
except:
    print('not opened')
    database = {}


name = input('Please enter you name: ')

if name not in database:
    database[name] = []


if not cap.isOpened():
    print('cannot open camera, not accessible')
    exit()

cap_count = 0
max_count = 20
last_embedding= None

def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 60

while cap_count < max_count:
    ret, frame = cap.read()

    if not ret:
        break
    
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    display = frame.copy()
    face_tensor = None
    boxes, _ = mtcnn.detect(rgb_frame)


    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int,box)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 100), 2)

        face_crop = rgb_frame[y1:y2,x1:x2]
        if face_crop.size != 0:
            face_tensor = mtcnn(face_crop)

    if face_tensor is not None:
        cv2.putText(display, f"captured:{cap_count}/{max_count}",
                    (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,100), 2)


    cv2.imshow('Registered Face',display)


    key = cv2.waitKey(1) & 0XFF

    if key == ord('c') and face_tensor is not None:

        if is_blurry(face_crop):
            print("Picture is blurry")
            continue

        embedding = resnet(face_tensor.unsqueeze(0)).detach().numpy()
        similar = False
        if last_embedding is not None:
            for vector in database[name]:
                similarity = cosine_similarity(embedding, vector)[0][0]
                if similarity > 0.90:
                    print('picture is similar, change the frame')
                    similar = True
                    break
        if not similar:
            database[name].append(embedding)
            last_embedding = embedding
            cap_count += 1

    elif key == ord('q'):
        exit()

cap.release()
cv2.destroyAllWindows()
    

# save embedding
try:
    with open(path,'wb') as f:
        pkl.dump(database, f)
except:
    print('database not opened')
print('Registration complete')