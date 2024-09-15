"""
using opencv V4.0.0 and python 3.6

several usfull functions i use in for the tracker and for the video and gif creator

"""

import cv2
import os
import traceback
import time
from stat import ST_MTIME

def get_ROI(img):
    print("\nuse `space` or `enter` to finish selection")
    print("use `c`     or `Esc`   to cancel selection (function will return zero by zero)")
    cv2.namedWindow("SELECT ROI", cv2.WINDOW_NORMAL)
    ## we can move the window into view - usfull in case it opens in a bad location
#    cv2.moveWindow("SELECT ROI",550,550)
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
    x,y - starting coordinate
    h,w - hight and width
    name - the window name
    """
    crop_img = img[y:y+h, x:x+w]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, crop_img)
    cv2.waitKey(1)
    return crop_img


def get_orig_path(file_path):
    """
    !!!kind of very specific for myself right now...!!!

    will only work if both img_directories are in same directroy themselves
    "...\8\croped"
    "...\8\origin"
    """
    try:
        base_folder = file_path.rsplit("\\",2)[0]
        cur_file_name = file_path.rsplit("\\",1)[1]
        file_numb = cur_file_name.split("_")[0]
        full_or_file_path = base_folder + "\\origin\\DSC_"+ file_numb + ".jpg"

    except IndexError as err:
        traceback.print_tb(err.__traceback__)
        print(err)
        print("check the names of the folders")
        return "ERROR"
    except FileNotFoundError as err:
        traceback.print_tb(err.__traceback__)
        print(err)
        print("check the names of the folders")
        return "ERROR"
    return full_or_file_path


def get_time(file_path):
    """
    still a bit shaky - ment to get the creation time from the output of the
    labs nikon camera files - not for raspberry PI cams/webcams/other formats
    """
    if "CROPED" in file_path:
        file_path = get_orig_path(file_path)
        if file_path == "EEROR":
            print("Croped in file name!")
            return("path time not found")
    STAT = os.stat(file_path)
    try:
        creat_time = time.strftime('%d-%m %H:%M', time.localtime(STAT[ST_MTIME]))
    except:
        print("there was an error getting the creation time")
        return("path time not found")
    return creat_time


def get_time_delta(str_img_time, cur_img_time):
    """converts both to time_since_epoch and the converts back to string in 00:00 H form"""
    # convert start
    if str_img_time == "path time not found":
        return "time_error"
    start_time = time.strptime(str_img_time+" -2000", '%d-%m %H:%M -%Y') # year added cause default dosn't work
    epoch_start = time.mktime(start_time)

    # convert curr_img
    curr_time = time.strptime(cur_img_time+" -2000", '%d-%m %H:%M -%Y') # year added cause default dosn't work
    epoch_curr = time.mktime(curr_time)

    # get delta and convert back to string
    delta = epoch_curr - epoch_start

    H = str(int(delta/60/60))
    M = int(delta/60 % 60)
    if M<10:
        M = "0" + str(M)
    else:
        M = str(M)
    delt_str = H+":"+M
#    else:
#        delt_str = time.strftime('%H:%M',time.localtime(delta))
    return delt_str


def text_on_img(img, text):
    """
    adds text to an img, tring to adjust the text size to the size of the img
    """
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
    ## where does the text go
    init_x_pos = int(img.shape[1]/8)
    init_y_pos = int(img.shape[0]/6)
    ## in case of multi line:
    text = text.split("\n")
    text_hight = cv2.getTextSize(text[0], font, font_size, font_thickness)[0][1]
    y_gap = int(text_hight*1.5)
    for i,line in enumerate(text):
        pos = (init_x_pos, init_y_pos+y_gap*i)
        img_with_text = cv2.putText(img, line, pos, font, font_size,(90,255,30),font_thickness)
    return img_with_text


def resize_img(img, scale_percent):
    """
    img - the image we want to resize
    scale_percent - number between 0-1
    """
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim)
    return resized
