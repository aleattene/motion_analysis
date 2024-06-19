import cv2
import mediapipe as mp
import numpy as np
import os
from motion_analysis.utils.angles import calculate_rotation_angle


# Function to analyze the source (stream or video)
def get_right_arm_rotation_angle(source):
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Errore nell'apertura del file video o della webcam: {source}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.splitext(source)[0] + '_analyzed.mp4'
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    initial_shoulder = None
    initial_hand = None
    max_rotation_angle = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fine del flusso video.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hand_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                if initial_shoulder is None and initial_hand is None:
                    initial_shoulder = shoulder_right
                    initial_hand = hand_right

                #
                # Calcola l'angolo di rotazione del braccio destro rispetto alla posizione iniziale
                angle_rotation = calculate_rotation_angle(initial_shoulder, initial_hand, shoulder_right, hand_right)

                # Aggiorna l'angolo massimo di rotazione
                if angle_rotation > max_rotation_angle:
                    max_rotation_angle = angle_rotation

                print(f"Angle Rotation: {angle_rotation}, Max Rotation: {max_rotation_angle}")

                # Disegna solo i segmenti richiesti
                cv2.line(image, tuple(np.multiply(shoulder_right, [image.shape[1], image.shape[0]]).astype(int)),
                         tuple(np.multiply(hand_right, [image.shape[1], image.shape[0]]).astype(int)),
                         (0, 255, 0), 2)
                cv2.circle(image, tuple(np.multiply(shoulder_right, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)
                cv2.circle(image, tuple(np.multiply(hand_right, [image.shape[1], image.shape[0]]).astype(int)), 5,
                           (0, 0, 255), -1)

                # Configura il riquadro di stato con l'angolo
                rect_color = (0, 128, 0) if abs(angle_rotation) < 45 else (245, 117, 16)

                cv2.rectangle(image, (0, 370), (170, 270), rect_color, -1)
                cv2.putText(image, f"{str(int(max_rotation_angle))}",
                            (20, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Scrivi il frame nel file
                out.write(image)

                # Visualizza l'angolo sul video (webcam o file)
                cv2.imshow('Angle Reveal', image)

            except Exception as e:
                print(f"An error occurred while processing the frame and calculating the angle: {e}")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Return the angle found during the video analysis
        # result_one = f"Spalla-Anca-Ginocchio (gradi): {str(int(angle_one))}"
        # result_two = f"Gomito-Spalla-Anca (gradi): {str(int(angle_two))}"

    # return result_one, result_two
