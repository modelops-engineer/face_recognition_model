1. Performance
    Webcam lag observed on CPU

Plan: reduce resolution, detect every N frames, add face tracking later

2. Data Quality
    same name people will have only one dataset, indexing requried
    Blur detection added using Laplacian variance

Plan: face-crop blur check instead of full-frame
      vector db

3. Recognition Accuracy
    it should check if face is blur or not, now it is taking whole frame and checking
    Using last embedding to detect similarity but it should check all from database
    the face tensor is matching with records when pressed C, real time check needs to be implemented
    Using multiple embeddings per person

Plan: threshold tuning using FAR/FRR
      add check with old saved data

4. Scalability
    MTCNN crash when face too close to camera
    Currently using pickle

Plan: move to FAISS / vector DB

5. Deployment
    Runs locally on CPU

Plan: ONNX export for faster inference