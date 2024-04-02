import cv2

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('cascade/red_ball.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the processed frame in a window
    cv2.imshow('Face Detection', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
