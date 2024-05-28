import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate the angle
def calculate_angle(a, b, c):
    a = np.array(a)  # Start point
    b = np.array(b)  # Intermediate point
    c = np.array(c)  # End point

    # Vectors
    ba = a - b
    bc = c - b

    # Scalar product and vector norm
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Calculate the angle between the two vectors (in degrees)
    cos_angle = dot_product / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle


# Function to analize the source (stream or video)
def analyze_video(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Errore nell'apertura del file video o della webcam: {source}")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    output_file = os.path.splitext(source)[0] + '_analyzed.avi'
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Initialize variable to store the angle offset
    angle_offset = None

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
                # shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                #             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                # elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                #          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                # wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                #          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Print coordinates for debugging
                # print(f"Hip: {shoulder}, Knee: {hip}, Ankle: {knee}")

                # Draw only the required segments
                cv2.line(image, tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.line(image, tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.circle(image, tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)

                # Calculate angle
                angle_revealed = calculate_angle(shoulder, hip, knee)
                if angle_offset is None:
                    angle_offset = 180 - angle_revealed

                angle_revealed = angle_revealed + angle_offset

                if angle_revealed < angle:
                    angle = angle_revealed

            except:
                print("An error occurred while processing the frame and calculating the angle.")

            # Choose the color of the rectangle based on the angle
            if angle < 90:
                rect_color = (0, 128, 0)  # Green
            else:
                rect_color = (245, 117, 16)  # Original color (light blue)

            # Setup status box with angle
            cv2.rectangle(image, (0, 0), (225, 73), rect_color, -1)
            cv2.putText(image, f"{str(int(angle))}",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Write the frame into the file
            out.write(image)

            # Render detections
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            #                           )

            # Display the angle on the video (webcam or file)
            cv2.imshow('Angle Reveal', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Return the angle found during the video analysis
        result = f"ANGLE FOUND (grade): {str(int(angle))}"

        # Convert the video to a format compatible with WhatsApp (using FFmpeg)
        converted_output_file = os.path.splitext(output_file)[0] + '_whatsapp.mp4'
        ffmpeg_command = [
            'ffmpeg',
            '-i', output_file,
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-strict', '-2',
            '-b:v', '1000k',
            '-b:a', '128k',
            converted_output_file
        ]
        subprocess.run(ffmpeg_command)

        return result


if __name__ == "__main__":
    # Ask the user if he wants to use the webcam or a video file
    # source_choice = input("Vuoi utilizzare la webcam o un file video? (webcam/file): ").strip().lower()
    # if source_choice == 'webcam':
    #     source = 0
    # else:
    #     source = input("Inserisci il percorso del file video: ").strip()
    source = "./video/01.mp4"

    print(analyze_video(source))
