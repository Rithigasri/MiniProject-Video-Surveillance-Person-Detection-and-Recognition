# VIDEO SURVEILLANCE PERSON DETECTION AND RECOGNITION

## ABSTRACT:
This project addresses a critical challenge in CCTV video surveillance: the need for automated person tagging and recognition. Conventional systems rely on manual efforts, leading to labor-intensive and error-prone processes. This project seeks to revolutionize video surveillance by using image processing techniques to automate person tagging and recognition, enhancing security and situational awareness.<br/>

By accurately tagging individuals, recognizing them and generating alerts, it empowers security professionals and emphasizes responsible surveillance practices. Through systematic data collection and advanced face detection and recognition techniques, this project lays the foundation for efficient and accurate video surveillance, promising improved security and responsive law enforcement.
## FEATURES:
<b>Facial Recognition:</b> 
Implementing facial recognition models like dlib to identify individuals in video frames.

<b>Clothing Matching:</b> 
Utilizing clothing matching techniques like color histograms to further validate identified individuals.

<b>Automated Tagging:</b> 
Developing algorithms to automatically tag recognized individuals in video frames using computer vision.

<b>Threshold Definition:</b>
Setting and fine-tuning thresholds for face recognition and clothing matching to ensure accurate identifications.

<b>Efficiency Optimization:</b> 
Streamlining algorithms to ensure efficient processing of video frames.

<b>Output Handling:</b> 
Managing the output of matched frames, preventing duplication or flooding of the output directory.

## REQUIREMENTS:
* Hardware – PCs
* Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
* Required Python packages- Opencv-python, dlib,numpy

## ARCHITECTURAL FLOW DIAGRAM:
### Overall Architecutre Diagram:
![Overall](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/8b458c89-c981-420a-9852-c7c601212489)

### Face Detection and Recognition - Architecture:
![Architecture](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/12799531-5b65-4c33-819a-b6060e008488)

### Clothing Color Analysis - Architecture:
![image](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/6a7dab30-9534-4877-9c83-f28b7512b3da)

## INSTALLATION:
1. Clone the repository:
```
git clone https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition.git
```
3. Install the required packages:
```
pip install opencv-python
pip install dlib
pip install numpy
```
4. Download and gather required files.
* Input Files:
  - v1.mov: The video file containing the scenes you want to process.
  - inputperson1.png: Image of the person whose face and clothing will be matched.
* Model Files:
  - dlib_face_recognition_resnet_model_v1.dat: Face recognition model file for dlib.
  - shape_predictor_68_face_landmarks.dat: Shape predictor model for dlib.
  - yolov3.weights and yolov3.cfg: YOLO model files for object detection.
  - coco.names: YOLO class names file.
4. Modify the file path:
Replace the file paths with the actual paths to your files on your computer.
5. Define folder to store the matched frames in your directory.
6. Step 5: Run the Code
Run the Python script in your terminal or command prompt:
```
python your_script_name.py
```
7. Follow the same process for different persons and optimize thresholds according to your specific application.

## PROGRAM:
```PYTHONimport cv2
import os
import dlib
import numpy as np

# Set the paths for the input files and the output directory
video_file = 'Output/v5.mov'
input_face_image = 'Output/inputperson1.png'
output_dir = "Output/Result"
os.makedirs(output_dir, exist_ok=True)

# Initialize face recognition models
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
input_image = cv2.imread(input_face_image)
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_faces = face_detector(input_image_rgb)

if len(input_faces) == 0:
    print("No faces found in the input image.")
else:
    landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")(input_image_rgb, input_faces[0])
    input_face_descriptor = face_rec_model.compute_face_descriptor(input_image_rgb, landmarks)

# Clothing matching using YOLO object detection
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')
output_layers = net.getUnconnectedOutLayersNames()

# Calculate histogram for input image
input_hist = cv2.calcHist([input_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
input_hist = cv2.normalize(input_hist, input_hist).flatten()

# Open video file for processing
video_capture = cv2.VideoCapture(video_file)
frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_count = 0
frame_number=0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_rate == 0:
        frame_name = f"{output_dir}/frame_{frame_number:04d}.jpg"
        frame_number += 1
        # Face recognition part
        frame_faces = face_detector(frame)
        for face in frame_faces:
            landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")(frame, face)
            frame_descriptor = face_rec_model.compute_face_descriptor(frame, landmarks)
            distance = np.linalg.norm(np.array(input_face_descriptor) - np.array(frame_descriptor))

            # YOLO object detection for clothing matching
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to 'person'
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()

                        # Clothing color analysis
                        person_roi = frame[y:y + h, x:x + w]
                        person_hist = cv2.calcHist([person_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        person_hist = cv2.normalize(person_hist, person_hist).flatten()

                        bhattacharyya_distance = cv2.compareHist(input_hist, person_hist, cv2.HISTCMP_BHATTACHARYYA)
                        threshold = 0.4

                        if distance < 0.4 or bhattacharyya_distance < threshold:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 8)
                            cv2.imwrite(frame_name, frame)
                            print(f"Match found in {frame_name}")

video_capture.release()
cv2.destroyAllWindows()


```
## OUTPUT:
* INPUT PERSON:
![inputperson1](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/22bc2250-d210-40db-9896-806b609498e7)
* MATCHING FRAMES [SAMPLE]:
![frame_0003](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/6ff51db0-30dc-4ff9-b857-cc15ec36a0b2)
![frame_0004](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/63b7bddc-4f4b-4d2e-8cf3-4d6245c4692d)
![frame_0005](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/a3e24d95-4e1f-4585-9ab6-4cfcb513c1ce)
![frame_0135](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/7b3ce8c2-a0d5-4b96-afc3-1ba60fd2662e)
![frame_0163](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/c0fde9e9-bb10-48f6-98f8-a85b14de0d0f)
![frame_0164](https://github.com/Rithigasri/MiniProject-Video-Surveillance-Person-Detection-and-Recognition/assets/93427256/b758349a-bf06-4e9d-8f5e-54e432cd7d50)

## RESULT:
In conclusion, our project combines the power of face recognition using dlib and clothing color analysis with color histogram and Bhattacharyya distance to provide a comprehensive solution for recognizing a person in a given video. By utilizing dlib, we can accurately identify and track individuals by their facial features. Additionally, the incorporation of clothing color analysis using color histograms and Bhattacharyya distance allows us to further enhance recognition accuracy.<br/>
This multi-modal approach not only identifies individuals by their facial characteristics but also considers their clothing, making it a robust and versatile solution for person recognition in video surveillance.However, there are limitations such as potential inaccuracies in case of occlusions, changes in lighting, or similar clothing colors among individuals. Despite these challenges, our system demonstrates the potential for improved video surveillance and identification in real-world scenarios.

