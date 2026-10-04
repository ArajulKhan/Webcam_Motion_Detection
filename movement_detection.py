import cv2
from winotify import Notification


# --------------------------------------------------
# FUNCTION: Send Windows notification
# --------------------------------------------------
def send_notification():
    toast = Notification(
        app_id="OpenCV Motion Detector",
        title="⚠ Motion Detected!",
        msg="Movement has been detected by your webcam."
    )

    toast.show()


# --------------------------------------------------
# OPEN WEBCAM
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# --------------------------------------------------
# READ FIRST FRAME
# --------------------------------------------------
ret, previous_frame = cap.read()

if not ret:
    print("Error: Could not read webcam.")
    cap.release()
    exit()


# Convert first frame to grayscale
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Blur the image to reduce small/noisy changes
previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)


# This prevents continuous notifications
notification_sent = False


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while True:

    # Read current frame
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    # --------------------------------------------------
    # PREPROCESS CURRENT FRAME
    # --------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)


    # --------------------------------------------------
    # COMPARE PREVIOUS FRAME WITH CURRENT FRAME
    # --------------------------------------------------

    difference = cv2.absdiff(previous_frame, gray)


    # Convert difference to black/white
    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )


    # Make nearby white areas join together
    threshold = cv2.dilate(threshold, None, iterations=2)


    # --------------------------------------------------
    # FIND MOVING OBJECTS
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    movement_detected = False


    for contour in contours:

        # Ignore very small movements/noise
        area = cv2.contourArea(contour)

        if area < 1000:
            continue


        movement_detected = True


        # Get rectangle around moving object
        x, y, w, h = cv2.boundingRect(contour)


        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )


        # Display text
        cv2.putText(
            frame,
            "MOTION DETECTED",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------
    # SEND WINDOWS NOTIFICATION
    # --------------------------------------------------

    if movement_detected:

        cv2.putText(
            frame,
            "WARNING: MOVEMENT!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


        # Send notification only once
        if not notification_sent:
            send_notification()
            notification_sent = True


    else:

        cv2.putText(
            frame,
            "NO MOTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Allow another notification after movement stops
        notification_sent = False


    # --------------------------------------------------
    # SHOW WEBCAM
    # --------------------------------------------------

    cv2.imshow("Motion Detection", frame)


    # --------------------------------------------------
    # SHOW DIFFERENCE IMAGE
    # --------------------------------------------------

    cv2.imshow("Movement Mask", threshold)


    # UPDATE PREVIOUS FRAME
    

    previous_frame = gray


   

    if cv2.waitKey(25) & 0xFF == ord("p"):
        break




cap.release()

cv2.destroyAllWindows()