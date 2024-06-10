import cv2
import mediapipe as mp
import numpy as np
import os
from motion_analysis.utils.angles import calculate_angle_degrees


# Function to analyze the source (stream or video)
def get_elbow_shoulder_hip_knee_double_angles(source):

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
    angle_offset_one_right = None
    angle_offset_two_right = None
    angle_offset_one_left = None
    angle_offset_two_left = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        angle_one_right = 180
        angle_two_right = 180
        angle_one_left = 180
        angle_two_left = 180
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
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Print coordinates for debugging
                # print(f"Elbow: {shoulder}, Shoulder: {hip}, Hip: {knee}")

                # Draw only the required segments - RIGHT
                cv2.line(image, tuple(np.multiply(shoulder_right, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(elbow_right, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(shoulder_right, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(hip_right, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(hip_right, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(knee_right, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)

                cv2.circle(image, tuple(np.multiply(elbow_right, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(shoulder_right, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(hip_right, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(knee_right, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)

                # Draw only the required segments - LEFT
                cv2.line(image, tuple(np.multiply(shoulder_left, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(elbow_left, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(shoulder_left, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(hip_left, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(hip_left, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(knee_left, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)

                cv2.circle(image, tuple(np.multiply(elbow_left, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(shoulder_left, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(hip_left, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(knee_left, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)

                # Calculate angle
                angle_revealed_one_right = calculate_angle_degrees(elbow_right, shoulder_right, hip_right)
                angle_revealed_two_right = calculate_angle_degrees(shoulder_right, hip_right, knee_right)
                angle_revealed_one_left = calculate_angle_degrees(elbow_left, shoulder_left, hip_left)
                angle_revealed_two_left = calculate_angle_degrees(shoulder_left, hip_left, knee_left)

                if angle_offset_one_right is None:
                    angle_offset_one_right = 130 - angle_revealed_one_right
                if angle_offset_two_right is None:
                    angle_offset_two_right = 105 - angle_revealed_two_right
                if angle_offset_one_left is None:
                    angle_offset_one_left = 130 - angle_revealed_one_left
                if angle_offset_two_left is None:
                    angle_offset_two_left = 105 - angle_revealed_two_left

                angle_revealed_one_right += angle_offset_one_right
                angle_revealed_two_right += angle_offset_two_right
                angle_revealed_one_left += angle_offset_one_left
                angle_revealed_two_left += angle_offset_two_left

                if angle_revealed_one_right < angle_one_right:
                    angle_one_right = angle_revealed_one_right
                if angle_revealed_two_right < angle_two_right:
                    angle_two_right = angle_revealed_two_right
                if angle_revealed_one_left < angle_one_left:
                    angle_one_left = angle_revealed_one_left
                if angle_revealed_two_left < angle_two_left:
                    angle_two_left = angle_revealed_two_left

            except:
                print("An error occurred while processing the frame and calculating the angle.")

            # Choose the color of the rectangle based on the angle
            LIGHT_BLUE_COLOR = (245, 117, 16)
            GREEN_COLOR = (0, 0, 128)
            RED_COLOR = (0, 128, 0)

            if angle_one_right < 120:
                rect_color_one_right = GREEN_COLOR
            else:
                rect_color_one_right = LIGHT_BLUE_COLOR

            if angle_two_right < 90:
                rect_color_two_right = RED_COLOR
            else:
                rect_color_two_right = LIGHT_BLUE_COLOR

            if angle_one_left < 120:
                rect_color_one_left = GREEN_COLOR
            else:
                rect_color_one_left = LIGHT_BLUE_COLOR

            if angle_two_left < 90:
                rect_color_two_left = RED_COLOR
            else:
                rect_color_two_left = LIGHT_BLUE_COLOR

            # Setup status box with angle
            cv2.rectangle(image, (0, 370), (170, 270), rect_color_one_right, -1)
            cv2.putText(image, f"{str(int(angle_one_right))}",
                        (20, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 550), (170, 450), rect_color_two_right, -1)
            cv2.putText(image, f"{str(int(angle_two_right))}",
                        (20, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            image_width = image.shape[1]
            extreme_right = image_width - 1
            print(f"Extreme right: {extreme_right}")

            cv2.rectangle(image, (479, 370), (309, 270), rect_color_one_left, -1)
            cv2.putText(image, f"{str(int(angle_one_left))}",
                        (329, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (479, 550), (309, 450), rect_color_two_left, -1)
            cv2.putText(image, f"{str(int(angle_two_left))}",
                        (329, 520),
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
        result_one_right = f"Spalla-Anca-Ginocchio - Lato Destro (gradi): {str(int(angle_one_right))}"
        result_two_right = f"Gomito-Spalla-Anca - Lato Destro (gradi): {str(int(angle_two_right))}"
        result_one_left = f"Spalla-Anca-Ginocchio - Lato Sinistro (gradi): {str(int(angle_one_left))}"
        result_two_left = f"Gomito-Spalla-Anca - Lato Sinistro (gradi): {str(int(angle_two_left))}"

    return result_one_right, result_two_right, result_one_left, result_two_left
