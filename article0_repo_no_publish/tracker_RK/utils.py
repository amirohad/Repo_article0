#%%
import os
import time
from datetime import datetime
import cv2
from stat import ST_MTIME
import piexif

global time_error
time_error = False


def get_ROI(img):
    print("\nuse `space` or `enter` to finish selection")
    print("use `c`     or `Esc`   to cancel selection (function will return zero by zero)")
    cv2.namedWindow("SELECT ROI", cv2.WINDOW_NORMAL)
#    cv2.moveWindow("SELECT ROI",550,550) #in case it opens in a bad location
    cv2.waitKey(1)
    bbox = cv2.selectROI("SELECT ROI", img, True)
    cv2.destroyWindow("SELECT ROI")
    return bbox

def get_multiple_ROIs(frame):
    """allows choosing multiple 'Regeions Of Interest' and returns them as list"""

    ## Creat a copy of the frame to show selected boxes
    show_selected_frame = frame.copy()

    ## init an empty list for the ROIs
    bboxs = []
    cv2.namedWindow("SELECT ROI", cv2.WINDOW_NORMAL)
    cv2.moveWindow("SELECT ROI", 50, 50)
    while True:
        cv2.namedWindow("SELECT ROI", cv2.WINDOW_NORMAL)

        bbox = cv2.selectROI("SELECT ROI", show_selected_frame, showCrosshair=True)
        if bbox == (0, 0, 0, 0):
            break
        bboxs.append(bbox)
        # drawing the selected boxes on the copy:
        for bbox in bboxs:
            show_selected_frame = cv2.rectangle(show_selected_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2],bbox[1]+ bbox[3]),
                                                (50, 50, 200) , 4)
            show_selected_frame = cv2.line(show_selected_frame, (bbox[0]+int(bbox[2]/2), bbox[1]), (bbox[0]+int(bbox[2]/2), bbox[1]+ bbox[3]),(50, 50, 200),2)
            show_selected_frame = cv2.line(show_selected_frame, (bbox[0], bbox[1]+int(bbox[3]/2)), (bbox[0]+bbox[2], bbox[1]+int(bbox[3]/2)),(50, 50, 200),2)

    cv2.destroyWindow("SELECT ROI")
    return bboxs

def crop(img, x, y, h, w, name):
    """
    shows the croped version of img, and returns it
    img - the image that you want to crop
    name - the window name
    """
    crop_img = img[y:y+h, x:x+w]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, crop_img)
    cv2.waitKey(1)
    return crop_img

def text_on_img(img, text, dx=0, dy=0):
    font = 5
    font_size = img.shape[0]/1000
    if img.shape[1]*1.5 < img.shape[0]:
        font_size = img.shape[1]/500
        font_thickness = 2
    font_thickness = 3
    if font_size>6:
        font_size = 6
    if font_size<1:
        font_size = 1
        font_thickness = 1
    init_x_pos = int(img.shape[1]/8)
    init_y_pos = int(img.shape[0]/6)
    # in case of multi line:
    text = text.split("\n")
    text_hight = cv2.getTextSize(text[0], font, font_size, font_thickness)[0][1]
    y_gap = int(text_hight*1.5)
    for i,line in enumerate(text):
        pos = (init_x_pos, init_y_pos+y_gap*i)
        img_with_text = cv2.putText(img, line, pos, font, font_size,(90,255,30),font_thickness)
    return img_with_text

def get_time_delta(strt_img_time, cur_img_time):
    """
    Converts a time string to a time delta string
    params:
        strt_img_time - the start img time string
        cur_img_time  - the current img time string
    returns:
        delt_str - the time delta string in the format HH:MM
    """

    # Check if the time strings are valid
    if len(strt_img_time) > 16 or len(cur_img_time) > 16 \
        or len(strt_img_time) == 0 or len(cur_img_time) == 0:
        global time_error 
        if not time_error:
            print(f"{50*'#'}\nno time data - probably not DSLR or cropped image\n{50*'#'}")
            time_error = True
        return "time_error"
    
    # Parse the time strings into datetime objects
    try: ## old version
        start_time = datetime.strptime(strt_img_time+" -2000", '%d-%m %H:%M -%Y')
        curr_time = datetime.strptime(cur_img_time+" -2000", '%d-%m %H:%M -%Y')
    except ValueError:
        start_time = datetime.strptime(strt_img_time, '%Y-%d-%m %H:%M')
        curr_time = datetime.strptime(cur_img_time, '%Y-%d-%m %H:%M')
    # Calculate the time delta in seconds
    delta = int((curr_time - start_time).total_seconds())

    # Convert the time delta to hours and minutes
    H = delta // 3600
    M = (delta % 3600) // 60

    # Format the time delta as a string
    delt_str = f"{H:02d}:{M:02d}"

    return delt_str

## new version of get_time
def get_time(img_file_path):
    """
    returns the time of image capture (when using DSLR!)
    """
    # try:
    piexif_data = piexif.load(img_file_path)
    if piexif_data["Exif"] == {}:
        global time_error 
        if not time_error:
            print(f"{50*'#'}\nno time data - probably not DSLR or cropped image\n{50*'#'}")
            time_error = True
        return ""
    img_time = piexif_data["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("utf-8")
    # convert time to the right format
    img_time = time.strptime(img_time, '%Y:%m:%d %H:%M:%S')
    img_time = time.strftime('%Y-%d-%m %H:%M', img_time)
    return img_time
    # except:
    #     return None
    
########## old 

def get_time_old_version(file_path):
    if "CROPED" in file_path.upper():
        file_path = get_orig_path(file_path)
        if file_path == "ERROR":
            return("path time not found")

    if not os.path.isfile(file_path):
        return("path time not found")
    STAT = os.stat(file_path)
    return time.strftime('%d-%m %H:%M', time.localtime(STAT[ST_MTIME]))

#%% testing 
    
# img_file_path = r"I:\DCIM\100ND780\DSC_5727.JPG"

# img_file_path = r"C:\Users\Roni\Desktop\for other ppl\agueda\time_series\4\Experiment-11.jpg"

# time1 = get_time(img_file_path)
# # time2 = get_time_new(path.replace("2", "3"))

# get_time_delta(time1, time1)
print(time_error)