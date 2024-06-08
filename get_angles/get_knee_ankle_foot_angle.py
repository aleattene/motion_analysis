import cv2
import mediapipe as mp
import numpy as np
import os
from motion_analysis.utils.angles import calculate_angle_degrees


# Function to analyze the source (stream or video)
def get_knee_ankle_foot_angle(source):
    # mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(source)

    # Check if the webcam or video file was opened successfully
    if not cap.isOpened():
        print(f"Errore nell'apertura del file video o della webcam: {source}")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    output_file = os.path.splitext(source)[0] + '_analyzed.avi'
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Initialize variable to store the angle offset
    angle_offset_one = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        angle_one = 130
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fine del flusso video.")
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
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                # Print coordinates for debugging
                # print(f"Elbow: {shoulder}, Shoulder: {hip}, Hip: {knee}")

                # Draw only the required segments
                cv2.line(image, tuple(np.multiply(ankle, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(ankle, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(foot, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)

                cv2.circle(image, tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(ankle, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(foot, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)

                # Calculate angle
                angle_revealed_one = calculate_angle_degrees(knee, ankle, foot)
                if angle_offset_one is None:
                    angle_offset_one = 130 - angle_revealed_one

                angle_revealed_one += angle_offset_one

                if angle_revealed_one < angle_one:
                    angle_one = angle_revealed_one

            except:
                print("An error occurred while processing the frame and calculating the angle.")

            # Choose the color of the rectangle based on the angle
            LIGHT_BLUE_COLOR = (245, 117, 16)
            GREEN_COLOR = (0, 128, 0)

            if angle_one < 170:
                rect_color_one = GREEN_COLOR  # Green
            else:
                rect_color_one = LIGHT_BLUE_COLOR

            # Setup status box with angle
            cv2.rectangle(image, (0, 700), (170, 600), rect_color_one, -1)
            cv2.putText(image, f"{str(int(angle_one))}",
                        (20, 670),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Write the frame into the file
            out.write(image)

            # Display the angle on the video (webcam or file)
            cv2.imshow('Angle Reveal', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Return the angle found during the video analysis
        result_one = f"Ginocchio-Caviglia-Piede (gradi): {str(int(angle_one))}"

    return result_one
