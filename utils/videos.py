import os
import subprocess


# Convert the video to a format compatible with WhatsApp (using FFmpeg)
def convert_video_to_whatsapp_compatibility(filename: str):
    converted_output_file = os.path.splitext(filename)[0] + '_whatsapp.mp4'
    ffmpeg_command = [
        'ffmpeg',
        '-i', filename,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-strict', '-2',
        '-b:v', '1000k',
        '-b:a', '128k',
        converted_output_file
    ]
    subprocess.run(ffmpeg_command)
