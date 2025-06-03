import cv2
import os

def save_video_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} frames to {output_folder}")

# Example usage
video_path = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Measurements\74\Side\CSRT_074_1\1.avi"
output_folder = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2_Pendulum\Measurements\74\Side\1_vid2frames"
save_video_frames(video_path, output_folder)