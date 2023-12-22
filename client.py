import cv2
from mtcnn import MTCNN

# Initialize the detector
detector = MTCNN()

# Capture video from the camera
cap = cv2.VideoCapture(0)


ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # You can adjust this threshold based on your camera's field of view
            motion_detected = True
            break

    if motion_detected:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in the frame
            results = detector.detect_faces(frame)

            # Draw rectangle around the faces and crop
            for result in results:
                x, y, width, height = result['box']
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 155, 255), 2)
                face = frame[y:y + height, x:x + width]
                cv2.imshow('Face', face)

            # Display the frame
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






cap.release()
cv2.destroyAllWindows()
