import cv2
import numpy as np
import os

def train_faces(data_dir, trained_model_path):
    # Create a face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Create a face cascade classifier
    face_cascade = cv2.CascadeClassifier('/home/max/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')

    # Create an instance of the pre-trained face mask detection model
    mask_detector = cv2.dnn.readNetFromCaffe(
        '/home/max/opencv-4.x/data/dnn/face_detector/deploy.prototxt',
        '/home/max/opencv-4.x/data/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    )

    # Create empty lists to store face samples and corresponding labels
    samples = []
    labels = []

    # Prompt the user to enter the name of the person to train
    person_name = input("Enter the name of the person to train: ")

    # Create a subdirectory for the person in the data directory
    person_dir = os.path.join(data_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # Set the IP camera URL
    ip_camera_url = 'https://192.168.100.57:8080'

    # Create a video capture object for the IP camera
    capture = cv2.VideoCapture(ip_camera_url)

    # Set a counter for image naming
    counter = 0

    while True:
        # Capture a frame
        ret, frame = capture.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region of interest
            face_roi = gray[y:y + h, x:x + w]

            # Detect mask on the face
            face_blob = cv2.dnn.blobFromImage(frame[y:y + h, x:x + w], 1.0, (300, 300), (104.0, 177.0, 123.0))
            mask_detector.setInput(face_blob)
            mask_detections = mask_detector.forward()

            # Check if a black mask is present
            mask_label = 0  # Label for black mask
            if mask_detections[0, 0, 0, 2] > 0.5:
                mask_label = 1  # Label for no mask

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the face image in the person's subdirectory
            image_name = f"image_{counter}.jpg"
            image_path = os.path.join(person_dir, image_name)
            cv2.imwrite(image_path, face_roi)

            # Add the face sample and corresponding label to the lists
            samples.append(face_roi)
            labels.append(mask_label)  # Use the mask label as the label

            # Increment the counter
            counter += 1


              # Display the frame
        cv2.imshow("Capture Training Images", frame)

        # Break if the user presses ESC or captures a sufficient number of images
        if cv2.waitKey(1) == 27 or counter >= 10:
            break

    # Release the video capture object
    capture.release()

    # Destroy all windows
    cv2.destroyAllWindows()

    # Convert the labels array to integer labels
    labels = np.array(labels, dtype=np.int32)

    # Train the face recognizer using the collected samples and labels
    face_recognizer.train(samples, labels)

    # Save the trained model
    face_recognizer.save(trained_model_path)


# Set the directory path for training data and the path to save the trained model
data_dir = '/home/max/Desktop/FaceTraining/data_dir'
trained_model_path = '/home/max/Desktop/FaceTraining/face_model.xml'

# Call the train_faces function
train_faces(data_dir, trained_model_path)
