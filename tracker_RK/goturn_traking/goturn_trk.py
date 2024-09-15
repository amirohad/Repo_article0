#%%
import cv2
import sys
import os

def get_init_roi(img_path):
    img = cv2.imread(img_path)
    ## resize the image if it is too big
    if img.shape[0]>1000 or img.shape[1]>1000:
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi

def run_tracker(bbox, frames_folder, frame_files, first_img_path):
    first_frame = cv2.imread(first_img_path)
    ## resize the image if it is too big
    if first_frame.shape[0]>1000 or first_frame.shape[1]>1000:
        first_frame = cv2.resize(first_frame, (0,0), fx=0.5, fy=0.5)
    
    # Initialize the GOTURN tracker
    tracker = cv2.TrackerGOTURN_create()
    ok = tracker.init(first_frame, bbox)
    
    ## Init the window 
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 600, 600)

    # Loop over all frames and update the tracker
    for file in frame_files[1:]:
        frame = cv2.imread(os.path.join(frames_folder, file))
        
        ## resize the image if it is too big
        if frame.shape[0]>1000 or frame.shape[1]>1000:
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        # Update tracker
        ok, bbox = tracker.update(frame)

        if ok:
            # Tracking success, draw the tracked object
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        # Display result
        cv2.imshow("Tracking", frame)
        
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()


cwd = os.getcwd()
cwd_files_lst =os.listdir(cwd)
if "goturn.prototxt" not in cwd_files_lst or "goturn.caffemodel" not in cwd_files_lst:
    print("Error: 'goturn.prototxt' OR 'goturn.caffemodel' not found in current directory!")
    ## If this error occoured you may be missong the files here:
    ## or you may not have them in the right directory... 
    # https://learnopencv.com/goturn-deep-learning-based-object-tracking/
    # https://github.com/spmallick/goturn-files
    sys.exit(1)

video_or_folder_name = r"C:\Users\Roni\Desktop\transfer_folder_SPC\3D_lights\top_croped\231121\top\Croped_2"
files_list = os.listdir(video_or_folder_name)
files_list = [f for f in files_list if f.lower().endswith(".jpg")]
first_img_path = os.path.join(video_or_folder_name, files_list[0])
bbox = get_init_roi(first_img_path)




run_tracker(bbox, video_or_folder_name, files_list, first_img_path)
#%%
## print cwd 

