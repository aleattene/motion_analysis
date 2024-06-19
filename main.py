import os
from dotenv import load_dotenv
from motion_analysis.get_angles.get_shoulder_hip_knee_ankles_angles import get_shoulder_hip_knee_ankle_angles
from motion_analysis.get_angles.get_elbow_shoulder_hip_knee_angles import get_elbow_shoulder_hip_knee_angles
from motion_analysis.get_angles.get_knee_ankle_foot_angle import get_knee_ankle_foot_angle
from motion_analysis.get_angles.get_elbow_shoulder90_hip_knee_angles import get_elbow_shoulder90_hip_knee_angles
from motion_analysis.get_angles.get_elbow_wrist_pinky_angle import get_elbow_wrist_pinky_angle
from motion_analysis.utils.videos import convert_video_to_whatsapp_compatibility
from motion_analysis.get_angles.get_elbow_shoulder_hip_knee_reverse_angles \
    import get_elbow_shoulder_hip_knee_reverse_angles
from motion_analysis.get_angles.get_elbow_shoulder_hip_knee_double_angles \
    import get_elbow_shoulder_hip_knee_double_angles
# from motion_analysis.get_angles.get_body_rotation_angle import get_right_arm_rotation_angle


def main():
    # Ask the user if he wants to use the webcam or a video file
    # source_choice = input("Vuoi utilizzare la webcam o un file video? (webcam/file): ").strip().lower()
    # if source_choice == 'webcam':
    #     source = 0
    # else:
    #     source = input("Inserisci il percorso del file video: ").strip()

    # Load environment variables
    # load_dotenv()
    # dir_videos = os.getenv('DIR_VIDEOS')
    webcam_source = 0
    result00 = get_elbow_shoulder_hip_knee_angles(webcam_source)
    print(result00)
    # source01 = dir_videos + os.getenv('VIDEO_01')
    # source02 = dir_videos + os.getenv('VIDEO_02')
    # source03 = dir_videos + os.getenv('VIDEO_03')
    # source04 = dir_videos + os.getenv('VIDEO_04')
    # source05 = dir_videos + os.getenv('VIDEO_05')
    # source06 = dir_videos + os.getenv('VIDEO_06')
    # source07 = dir_videos + os.getenv('VIDEO_07')
    # source08 = dir_videos + os.getenv('VIDEO_08')

    # postfix_analyzed = '_analyzed.avi'

    # result01 = get_elbow_shoulder_hip_knee_angles(source01)
    # print(result01)
    # convert_video_to_whatsapp_compatibility(source01[0:-4] + postfix_analyzed)

    # result02 = get_elbow_shoulder_hip_knee_double_angles(source02)
    # print(result02)
    # convert_video_to_whatsapp_compatibility(source02[0:-4] + postfix_analyzed)

    # result03 = get_shoulder_hip_knee_ankle_angles(source03)
    # print(result03)
    # convert_video_to_whatsapp_compatibility(source03[0:-4] + postfix_analyzed)

    # result04 = get_elbow_shoulder_hip_knee_reverse_angles(source04)
    # print(result04)
    # convert_video_to_whatsapp_compatibility(source04[0:-4] + postfix_analyzed)

    # result05 = get_right_arm_rotation_angle(source05)
    # print(result05)
    # convert_video_to_whatsapp_compatibility(source05[0:-4] + postfix_analyzed)

    # result06 = get_knee_ankle_foot_angle(source06)
    # print(result06)
    # convert_video_to_whatsapp_compatibility(source06[0:-4] + postfix_analyzed)

    # result07 = get_elbow_wrist_pinky_angle(source07)
    # print(result07)
    # convert_video_to_whatsapp_compatibility(source07[0:-4] + postfix_analyzed)

    # result08 = get_elbow_shoulder90_hip_knee_angles(source08)
    # print(result08)
    # convert_video_to_whatsapp_compatibility(source08[0:-4] + postfix_analyzed)


if __name__ == "__main__":
    main()
