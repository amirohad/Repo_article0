# 1. use:  "& c:/Users/Amir/Documents/PHD/Python/GitHub/Lab-toolbox/img_tools/tracker_RK/.conda/python.exe" as 'python'
# 2. install only opencv-contrib-python==4.5.4.60 (no opencv-python)
# 3. install numpy==1.21.6

import cv2 # V4.5.0
import sys
import os
import re
import utils
import manual_tracking
import time


if cv2.__version__ >= "4.5.4":
    new_cv = True
    print(f"{cv2.__version__=}")
else:
    new_cv = False


# Save tracked objects rois
def save_result_rois(output_folder, boxes, bbox_inits, tracker_types, init,
                       object_name, file_name,
                     output_path, first_tiral):
    # Create directory to save
    pic_time = utils.get_time(file_name)
    dirName = os.path.dirname(output_folder)
    videoName = os.path.splitext(os.path.basename(output_folder))
    dirName = dirName + "\\" + output_path
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    dirName = dirName + "\\Rois_" + videoName[0] + "_" + object_name
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    # Create files and save init box as first line
    if init:
        # Save trackers init roi
        for i in range(len(tracker_types)):
            # roi file path
            path_tracker_rois = dirName + "\\" + tracker_types[i] + '.txt'
            # create tracker file
            if first_tiral:
                my_file = open(path_tracker_rois, 'w+')
            else:
                my_file = open(path_tracker_rois, 'a')

            for j, bbox_init in enumerate(bbox_inits):
                print("i = {}, j = {}, bbox = {}".format(i,j,bbox_init))
                my_file.writelines(", ".join([str(s) for s in list(bbox_init)])+
                                   ", " + file_name + ", "+ pic_time + ", "
                                   + str(j) + ", Start \n")
            # close file
            my_file.close()
        return path_tracker_rois
    else: # (if not init)
        # Append files and save current box
        for i in range(len(tracker_types)):
            for j, new_box in enumerate(boxes):
                # roi file path
                path_tracker_rois = dirName + "\\" + tracker_types[i] + '.txt'
                with open(path_tracker_rois, "a") as my_file:
                    my_file.writelines(", ".join([str(round(s, 2)) for s in list(new_box)]) + ", " +
                                       file_name +", "+ pic_time + ", " + str(j) + ", Auto\n")
        return path_tracker_rois

# set video output for saving
def create_video_results(video_or_folder_name, frame,   object_name, output_path):
    dirName = os.path.dirname(video_or_folder_name)
    videoName = os.path.splitext(os.path.basename(video_or_folder_name))
    dirName = dirName + "\\" + output_path
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    save_video_name = dirName + "\\" + videoName[0] + object_name + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    frame_height, frame_width, layers = frame.shape
    video_out = cv2.VideoWriter(save_video_name, fourcc, 24.0, (frame_width, frame_height))
    print("video will be saved to: " + save_video_name)
    return video_out

# create list of colors to use to show trackers
def get_box_colors():
    colors = []
    #colors.append((0, 0, 0))  # Black
    #colors.append((255, 0, 0))  # Blue
    #colors.append((200, 200, 0))  # Cyan
    #colors.append((0, 200, 0))  # Lime
    
    colors.append((200, 200, 0))  # Cyan
    colors.append((255, 255, 0))  # Yellow
    colors.append((0, 0, 255))  # Red
    colors.append((0, 255, 255))  # Cyan
    colors.append((255, 0, 255))  # Magenta
    colors.append((192, 192, 192))  # Silver
    colors.append((128, 128, 128))  # Gray
    colors.append((128, 0, 0))  # Maroon
    colors.append((128, 128, 0))  # Olive
    colors.append((0, 128, 0))  # Green
    colors.append((128, 0, 128))  # Purple
    colors.append((0, 128, 128))  # Teal
    colors.append((0, 0, 128))  # Navy
    colors.append((188, 143, 143))  # rosy brown
    colors.append((230, 230, 250))  # lavender
    colors.append((244, 164, 96))  # sandy brown
    colors.append((0, 206, 209))  # dark turquoise
    colors.append((64, 224, 208))  # turquoise
    colors.append((72, 209, 204))  # turquoise pale
    colors.append((175, 238, 238))  # turquoise aqua
    colors.append((127, 255, 212))  # aqua marine
    colors.append((255, 0, 255))  # magenta / fuchsia
    colors.append((218, 112, 214))  # orchid
    colors.append((199, 21, 133))  # medium violet
    colors.append((219, 112, 147))  # pale violet
    colors.append((255, 20, 147))  # deep pink
    colors.append((255, 105, 180))  # hot pink
    return colors

# Return a list of all file names in path which are images
def list_images_in_path(directory):
    included_extensions = ['jpg', 'bmp', 'png', 'gif', 'jpeg']
    onlyfiles = [f for f in os.listdir(directory)
                 if any(f.lower().endswith(ext) for ext in included_extensions)]

    onlyfiles.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    full_path_files = [directory + '\\' + s for s in onlyfiles]

    return full_path_files

# Draw the detected bounding box and tracker name on a given frame
def draw_bounding_box(frame, boxes, tracker_ok, #ground_truth_bbox,
                      tracker_types, colors, i_frame, path_tracker_rois):

    color_i = 0

    # Draw tracked objects
    for i, new_box in enumerate(boxes):
        color_i = color_i + 1

        p1 = (int(new_box[0]), int(new_box[1]))
        p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))

        # cv2.rectangle(frame, p1, p2, colors[i + 2], 2, 4)
# =============================================================================
#         # show rectangle in another window
#         crop_img = frame[int(new_box[1]):int(new_box[1])+int(new_box[3]),
#                          int(new_box[0]):int(new_box[0])+int(new_box[2])]
#         cv2.namedWindow("trackedROI", cv2.WINDOW_NORMAL)
#         cv2.imshow("trackedROI", crop_img)
#         cv2.waitKey(1)
# =============================================================================
    # draw tracked path
    with open(path_tracker_rois, "r") as tracker_hist_data:

        for line in tracker_hist_data:
            num = int(line.split(",")[-2])
            ## need to change this to regex! - currently won't work if not under C directory...
            dots = (line.split(", C")[0]).split(", ")[:4]
            dots = list(map(float, dots))
            x = (int(dots[0] + dots[2]/2))
            y = (int(dots[1] + dots[3]/2))
            # auto in red
            if line.split(", ")[-1].strip() == "Auto":
                cv2.circle(frame, (x, y), 1, colors[num+2], -1)
            # manual in Orange
            elif line.split(", ")[-1].strip() == "Manual":
                cv2.circle(frame, (x, y), 1, (0,150,255), -1) #

    return frame

# Create tracker according to tracker_type
def create_tracker(tracker_type):
    if new_cv:
        cv = cv2.legacy
    else:
        cv = cv2
    tracker = []
    if tracker_type == 'MIL':
        try: tracker = cv.TrackerMIL_create()
        except : pass
    elif tracker_type == 'KCF':
        try: tracker = cv.TrackerKCF_create()
        except : pass
    elif tracker_type == 'TLD':
        try: tracker = cv.TrackerTLD_create()
        except : pass
    elif tracker_type == 'MEDIANFLOW':
        try: tracker = cv.TrackerMedianFlow_create()
        except : pass
    elif tracker_type == 'MOSSE':
        try: tracker = cv.TrackerMOSSE_create()
        except : pass
    elif tracker_type == "CSRT":
        try: tracker = cv.TrackerCSRT_create()
        except : pass
    return tracker

# return a frame and status, by file name/ video name and frame number
def load_image_from_file_or_video(run_images_from_folder, video_files, frame_number, video):
    if run_images_from_folder:
        if frame_number >= len(video_files):
            ok = False
            return ok, [], None
        filename = video_files[frame_number]
        # video_folder + str(frame_number) + ".jpg"
        if not os.path.isfile(filename):
            print('file not found: ' + filename)
            ok = False
            return ok, [], filename
        frame = cv2.imread(filename, cv2.IMREAD_COLOR)
        if frame is None:
            print('file not found: ' + filename)
            ok = False
            return ok, [], filename
        ok = True

    else:
        # video.set(cv2.cv2_CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
    return ok, frame, filename

## Return initial bounding box from selection
def get_initial_bounding_boxs(frame):
    ## Muliple trackers:
    ## Creat a copy of the frame to show selected bboxes
    show_selected_frame = frame.copy()
    bboxs = []
    while True:
        cv2.namedWindow("SELECT ROI", cv2.WINDOW_NORMAL)
        # print(cv2.getWindowProperty("SELECT ROI", cv2.WND_PROP_VISIBLE))
        # cv2.setWindowProperty("SELECT ROI", cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow("SELECT ROI", 600, 600)
        bbox = cv2.selectROI("SELECT ROI", show_selected_frame, showCrosshair=True)
        if bbox == (0, 0, 0, 0):
            break
        bboxs.append(bbox)
        # drawing the already selected boxes on the copy:
        for bbox in bboxs:
            x_0,y_0,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
            frame_color = (50, 50, 200)
            show_selected_frame = cv2.rectangle(show_selected_frame, (x_0, y_0), (x_0+w,y_0+h),frame_color , 2)
            show_selected_frame = cv2.line(show_selected_frame, (x_0+int(w/2), y_0), (x_0+int(w/2), y_0+ h),frame_color,2)
            show_selected_frame = cv2.line(show_selected_frame, (x_0, y_0+int(h/2)), (x_0+w, y_0+int(h/2)),frame_color,2)
    return bboxs

def tracker_loop(frame_to_start, run_images_from_folder, video_files, video,
                 multi_tracker,video_or_folder_name,
                 tracker_types, object_name, output_path, first_tiral,
                 # first_tiral is mine and for difrence between "a" and "w" writing!
                 video_out, colors = get_box_colors()):
    str_img_time = utils.get_time(video_files[0])
    # loop through frames
    i_frame = frame_to_start
    print(f"starting frame:{i_frame}")
    k = 0
    while True:
        try:
            ok, frame, f_name = load_image_from_file_or_video(run_images_from_folder, video_files, i_frame, video)
            # If the ok came back False, we didn't load an img
            if not ok:
                k = 27
                break
            
            # Get updated location of objects in subsequent frames
            ok, boxes = multi_tracker.update(frame)
            
            # If the tracking is lost - break
            if not ok:
                print("\nLost it at -", i_frame)
                return i_frame, 112

            # save current rois from all trackers to txt files
            path_tracker_rois = save_result_rois(video_or_folder_name, boxes, [],
                             tracker_types, False,   object_name,
                             f_name, output_path, first_tiral)

            ## Resize frame if necessary
            frame_resized = cv2.resize(frame, (int(frame.shape[0]/2), int(frame.shape[1]/2)))
            # frame_resized  = frame

            frame_with_box = draw_bounding_box(frame_resized, boxes, ok, #ground_truth_bbox,
                                              tracker_types, colors, i_frame,
                                              path_tracker_rois)

            t = utils.get_time(f_name)
            dt = utils.get_time_delta(str_img_time, t)
            utils.text_on_img(frame_with_box, dt)

            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.imshow("Tracking", frame_with_box)

            video_out.write(frame_with_box)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27: # Esc key
                print("\nCaught ESC")
                break
            if k == 112: # p key
                print("\nPaused, go do some manual work!\n last one was", i_frame)
                break

            i_frame += 1
        except KeyboardInterrupt:
            print("\nCaught Keyboard Interrupt")
            print(i_frame)
            k = 27
            break

    return i_frame, k

# Run tracker according to tracking params. Display result on images
def run_tracker_wrapper(tracker_types, run_images_from_folder,
                        video_or_folder_name, frame_to_start,
                        object_name, output_path, first_tiral):
    """ The function runs a few opencv2 trackers on a video
    (which has previously been split into frames)
        then it saves a video of the results

        Parameters
        ----------
        tracker_types : list
                    a list of the tracker types to run
                    the names should match the types specified in create_tracker funciton
                    example: ['MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
        run_images_from_folder : bool
                    if true run on frames what were extraacted from a video, if false run movie.
        video_or_folder_name : str
                    Path to a folder contains videos frames (assumed to be in YUV format).
                    example: 'C:\\ExamplesOfInputs4Testing\\vot2016Simple\\blanket_yuv'
        frame_to_start : int
                    frame to start tracking
                    example: 0
        object_name : str
                    name to add to end of outputs
                    example: 'obj1'
       """

    # Create MultiTracker object
    if not new_cv:
        multi_tracker = cv2.MultiTracker_create()
    else:
        multi_tracker = cv2.legacy.MultiTracker_create()
        
    # load first frame
    if run_images_from_folder:
        video = []
        video_files = list_images_in_path(video_or_folder_name)
        if video_files == []:
            print("bad file format for imgs in path!, no files found")
            sys.exit()
    else:
        video = cv2.VideoCapture(video_or_folder_name)
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
        run_images_from_folder = []
        video_files = []

    ok, frame, f_name = load_image_from_file_or_video(run_images_from_folder,
                                                      video_files, frame_to_start,
                                                      video)
    if not ok:
        print('Cannot read load_image_from_file_or_video at init')
        sys.exit()
# =============================================================================
#     # resize frame if necessary
#     shape_frame = frame.shape
#     if shape_frame[1] > 1920:
#         AR_frame = float(shape_frame[1]) / float(shape_frame[0])
#         width_frame = 1920
#         height_frame = round(float(width_frame) / AR_frame)
#     else:
#         width_frame = shape_frame[1]
#         height_frame = shape_frame[0]
# =============================================================================
    
    ## Get initial bounding box from user input
    init_bboxs = get_initial_bounding_boxs(frame)
    cv2.destroyAllWindows()

    ## If no ROIs were selected break off
    if len(init_bboxs) == 0:
        print("no ROI selected!")
        return

    ## Initialize MultiTracker
    for tracker_type in tracker_types:
        for bbox in init_bboxs:
            print(f"\n{tracker_type=}")
            multi_tracker.add(create_tracker(tracker_type), frame, bbox)

    # Set video output for saving
    video_out = create_video_results(video_or_folder_name, frame,   object_name, output_path)

    # Save current rois from all trackers to txt files
    path_tracker_rois = save_result_rois(video_or_folder_name, [], init_bboxs,
                                         tracker_types, True,   object_name,
                                         f_name, output_path, first_tiral)
    i_frame = frame_to_start
    while i_frame < len(video_files):
        # Auto tracking
        i_frame, k  = tracker_loop(i_frame, run_images_from_folder, video_files, video,
                                   multi_tracker,video_or_folder_name,
                                   tracker_types, object_name, output_path, first_tiral,
                                   video_out, colors = get_box_colors())
        # Esc
        if k == 27:
            break
        # Manual tracking
        elif k == 112: # p key
            break
            if run_images_from_folder == True:
                i_frame, manual_k = manual_tracking.loop_through_imgs(video_or_folder_name,
                                                                      path_tracker_rois, i_frame,
                                                                      video_out)
            else:
                print("Nope... can't track manualy with video")
                break
            # back to Auto tracking
            if manual_k == 27: #if Esc Key from the tracker continue Auto tracker

                multi_tracker = cv2.MultiTracker_create()

                ok, frame, f_name = load_image_from_file_or_video(run_images_from_folder,
                                                      video_files, i_frame,
                                                      video)
                # Get initial bounding box from user input
                init_bboxs = (frame)
                cv2.destroyAllWindows()

                ## If no ROIs were selected break off
                if len(init_bboxs) == 0:
                    print("no ROI selected!")
                    break

                # Re-Initialize MultiTracker
                for tracker_type in tracker_types:
                    for bbox in init_bboxs:
                        print(tracker_type)
                        multi_tracker.add(create_tracker(tracker_type), frame, bbox)

                path_tracker_rois = save_result_rois(video_or_folder_name, [], init_bboxs,
                                                     tracker_types, True,   object_name,
                                                     f_name, output_path, first_tiral = False)

    print(f'\nreached end of video files or stopped: frame_number={i_frame}')
    video_out.release()
    cv2.destroyAllWindows()
    if not run_images_from_folder:
        video.release()

        
def main():
    #%%
    
    run_images_from_folder = True
    frame_to_start = 0  
    first_tiral = True 

    # Choose which trackers to run
    tracker_type_list = ['CSRT']#, 'KCF']#, 'TLD', 'MEDIANFLOW','MIL', 'MOSSE']

    # for i in [1,2,3,4,5]:
    for i in [1]:
        
        elem = f"Croped_{i}"
        # video_or_folder_name = r"C:\Users\Roni\Desktop\transfer folder\3D lights\231227\top\{}".format(elem)
        video_or_folder_name = r"C:\Users\Amir\Documents\PHD\Experiments\Force Measurements\Exp2f_CN_stem_dynamics\1\test"
        print(video_or_folder_name)
    
        object_name = f"{i}"
        output_path = "\Tracked"

        outfolder = os.path.dirname(video_or_folder_name) + "\\" + output_path
    
        if os.path.isdir(outfolder):
           print("\nthat folder already exists! are you sure you want to continue?")
           ans = input("y/n? ")
           if ans == "n":
               sys.exit()

        print("\nstart: ")
        start = time.perf_counter()
        run_tracker_wrapper(tracker_type_list, run_images_from_folder,
                            video_or_folder_name, frame_to_start,
                            object_name, output_path,
                            first_tiral)
        ## print how long it took
        print(f"it took: {time.perf_counter() - start:.2f} [sec]")


if __name__ == "__main__":
    main()
