import cv2
import mediapipe as mp
import numpy as np
import os
from motion_analysis.utils.angles import calculate_angle_degrees


# Function to analyze the source (stream or video)
def get_elbow_shoulder_hip_knee_angles(source):

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
    angle_offset_two = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        angle_one = 180
        angle_two = 180
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
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Print coordinates for debugging
                # print(f"Elbow: {shoulder}, Shoulder: {hip}, Hip: {knee}")

                # Draw only the required segments
                cv2.line(image, tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)

                cv2.circle(image, tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)), 5,
                            (0, 0, 255), -1)

                # Calculate the angle
                angle_revealed_one = calculate_angle_degrees(elbow, shoulder, hip)
                angle_revealed_two = calculate_angle_degrees(shoulder, hip, knee)
                if angle_offset_one is None:
                    angle_offset_one = 180 - angle_revealed_one
                if angle_offset_two is None:
                    angle_offset_two = 180 - angle_revealed_two

                angle_revealed_one += angle_offset_one
                angle_revealed_two += angle_offset_two

                if angle_revealed_one < angle_one:
                    angle_one = angle_revealed_one
                if angle_revealed_two < angle_two:
                    angle_two = angle_revealed_two

            except:
                print("An error occurred while processing the frame and calculating the angle.")

            # Choose the color of the rectangle based on the angle
            LIGHT_BLUE_COLOR = (245, 117, 16)
            GREEN_COLOR = (0, 128, 0)
            RED_COLOR = (0, 0, 128)

            if angle_one < 170:
                rect_color_one = RED_COLOR
            else:
                rect_color_one = LIGHT_BLUE_COLOR

            if angle_two < 90:
                rect_color_two = GREEN_COLOR
            else:
                rect_color_two = LIGHT_BLUE_COLOR

            # Setup status box with angle
            cv2.rectangle(image, (0, 370), (170, 270), rect_color_one, -1)
            cv2.putText(image, f"{str(int(angle_one))}",
                        (20, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 550), (170, 450), rect_color_two, -1)
            cv2.putText(image, f"{str(int(angle_two))}",
                        (20, 520),
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
        result_one = f"Spalla-Anca-Ginocchio (gradi): {str(int(angle_one))}"
        result_two = f"Gomito-Spalla-Anca (gradi): {str(int(angle_two))}"

    return result_one, result_two
