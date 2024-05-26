import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to analize the source (stream or video)
def analyze_video(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Errore nell'apertura del file video o della webcam: {source}")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        angle = 180
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fine del flusso video o errore nella ricezione del frame.")
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle_revealed = calculate_angle(shoulder, elbow, wrist)
                if angle_revealed < angle:
                    angle = angle_revealed

            except:
                print("An error occurred while processing the frame and calculating the angle.")

            # Setup status box with angle
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, f"{str(int(angle))}",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # Display the angle on the video (webcam or file)
            cv2.imshow('Angle Reveal', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Return the angle found during the video analysis
        result = f"ANGLE FOUND (grade): {str(int(angle))}"
        return result


if __name__ == "__main__":
    # Ask the user if he wants to use the webcam or a video file
    source_choice = input("Vuoi utilizzare la webcam o un file video? (webcam/file): ").strip().lower()
    if source_choice == 'webcam':
        source = 0
    else:
        source = input("Inserisci il percorso del file video: ").strip()

    print(analyze_video(source))


