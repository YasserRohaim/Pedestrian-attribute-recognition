from deepface import DeepFace
import cv2
import time

# Setup
cap = cv2.VideoCapture(0)
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (0, 0))
frame_count = 0
analysis_interval = 100  # Analyze every 15 frames
resize_width = 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
   
    if frame_count % analysis_interval == 0:
        try:
            img_W = int(frame.shape[1])
            img_H = int(frame.shape[0])
            # Set input size
            detector.setInputSize((img_W, img_H))
            # Getting detections
            faces = detector.detect(frame)
            
            if len(faces):
                
                result = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)
                print("DeepFace result:", result, type(result))
                print(f"Age: {result[0]['age']}, Gender: {result[0]['gender']}")
        except Exception as e:
            print("DeepFace detection failed:", e)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
