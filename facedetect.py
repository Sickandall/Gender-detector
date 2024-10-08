import cv2
import numpy as np

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load gender classification model
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# List of gender classes
gender_list = ['Male', 'Female']

# Replace with your IP webcam URL
ip_webcam_url = "http://192.168.0.155:8080/video"
cap = cv2.VideoCapture(ip_webcam_url)
window_name = 'IP Webcam'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 360)

# List to store the last few gender predictions to smooth the result
gender_predictions = []
smooth_window_size = 5  # Number of frames to smooth over

while True:
    ret, frame = cap.read()
    if ret:
        # Convert frame to blob for face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Increased threshold for more accurate detections
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Add padding to face cropping to improve results
                padding = 10
                face = frame[max(0, y-padding):min(h, y1+padding), max(0, x-padding):min(w, x1+padding)]

                if face.size > 0:
                    # Resize face image to 227x227 as required by the gender model
                    face_resized = cv2.resize(face, (227, 227))

                    # Prepare the face for gender classification
                    blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()

                    # Append prediction to the smoothing window
                    gender_predictions.append(gender_preds[0].argmax())

                    # Limit the size of the smoothing window
                    if len(gender_predictions) > smooth_window_size:
                        gender_predictions.pop(0)

                    # Determine final gender by majority vote
                    gender = gender_list[max(set(gender_predictions), key=gender_predictions.count)]

                    # Draw the bounding box and label on the frame
                    label = f"{gender}"
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow(window_name, frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()

