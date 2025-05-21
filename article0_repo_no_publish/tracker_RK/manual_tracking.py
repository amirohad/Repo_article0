# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:25:05 2019

@author: YasmineMnb
"""

import os
import cv2
import numpy as np
import utils

def load_points_from_tracker_file(points_file):
    ## this needs to change to fit different locations!
    ## redo with regex!!!
    points_dct = {}
    with open (points_file, "r") as tracker_hist_data:
        for line in tracker_hist_data:
            dots = (line.split(", D")[0]).split(", ")
            dots = list(map(float, dots))
            # the tracker saves the rectangle and not the center dot
            x = (int(dots[0] + dots[2]/2))
            y = (int(dots[1] + dots[3]/2))
            file_name = line.split(",")[-2].split("\\")[-1]
            points_dct[file_name] = ((x,y),"Auto")
        # save a backup of the original points_file
        back_up_name = points_file.strip(".txt") + "_BACKUP.txt"
        with open (back_up_name, "w") as backup:
            backup.write(tracker_hist_data.read())
    return points_dct

def add_previous_points(points_dct, img):
    for i in sorted(points_dct):
        (x,y) = points_dct[i][0]
        cv2.circle(img, (x, y), 1, (0,150,255), -1)
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.imshow("Tracking", img)
    return img

def click_event(event, x, y, flags, param ):
    # what to do with which mous function -
    # param = [img_name, img, points, original_img]
    curr_pic_name = param[0]
    if event == cv2.EVENT_LBUTTONDOWN: ##left click:
        print(x,y)
        param[2][curr_pic_name] = ((x,y),"Manual")
        drawing_copy = param[1].copy()
        drawing_copy = cv2.circle(drawing_copy, (x, y), 1, (0, 255, 0), -1)
        cv2.imshow("Tracking", drawing_copy)
        cv2.waitKey(1)

    elif event == cv2.EVENT_RBUTTONDOWN: ## right click:
        print("clean")
        param[2][curr_pic_name] = (-1,-1)
        param[1] = param[3]
        cv2.imshow("Tracking", param[1])
        cv2.waitKey(1)

def clickable_pic_loop(img, img_name, i, points_dct, folder_path, video_out):
    original_img = cv2.imread(folder_path + "\\" + img_name)
    img = cv2.imread(folder_path + "\\" + img_name)
    img = add_previous_points(points_dct, img)
    cv2.imshow("Tracking", img)
    while True:
        cv2.setMouseCallback("Tracking", click_event, param = [img_name, img, points_dct, original_img])

        k=cv2.waitKey(1) & 0xFF
        if k==52: #4 key(numpad left)
            break
        elif k==54: #6 key(numpad right)
            break
        elif k==27: # Esc key
            break
        elif k==13: # Enter key (relevent for trackbar)
            break
    i = cv2.getTrackbarPos("trackbar", "Tracking")

    if video_out != None:
        video_out.write(img)

    return k, i

def trackbar_func(val):
    #can't escape an empty func for - createTrackbar
    pass


def save_added_points(points_dct, points_file, folder_path):
    """
    will only make changes to the points manualy added
    """
    line = ""
    for i in sorted(points_dct):

        if points_dct[i][1] == "Manual":
            point = points_dct[i][0]
            full_img_path = folder_path + "\\" + i
            box_size = 40
            box_x = point[0]-box_size/2
            box_y = point[1]-box_size/2
            img_time = utils.get_time(full_img_path)
            line = ', '.join(map(str, [box_x, box_y, box_size, box_size, full_img_path, img_time, -1, "Manual"]))
            with open(points_file, "a") as out_file:
                out_file.write(line)
                out_file.write("\n")


def loop_through_imgs(folder_path, points_file, i = 0, video_out = None):
    # list of imgs in the folder
    pic_lst = os.listdir(folder_path)
    # try loading exicting points from points_file
    try:
        points_dct = load_points_from_tracker_file(points_file)
    except FileNotFoundError:
        print("no such file?\n points are saved to empty dict")
        points_dct = {}
    # creat the window
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("trackbar", "Tracking" , 0, len(pic_lst), trackbar_func)
    # actual loop for the folder
    while i < len(pic_lst):
        img_name = pic_lst[i]
        img = cv2.imread(folder_path + "\\" + img_name)

        # update Trackbar and add points colected so far
        cv2.setTrackbarPos("trackbar", "Tracking", i)

        # open current img for clicking
        k,i = clickable_pic_loop(img, img_name, i, points_dct, folder_path, video_out)

        # how to continue:
        if k==27:
            print("you stoped the manual collection")
            break
        elif k==54: # 6 key(numpad right) (NEXT)
            i += 1
        elif k==52: # 4 key(numpad left) (BACK)
            i -= 1
        elif k==13: # Enter key (if used the trackbar)
            continue

    print("you should save to this file now:", points_file)
    save_added_points(points_dct, points_file, folder_path)
    return i, k




#%%

def main():
    """
    if only manual tracking should be used
    """

    folder_path = r"C:\Users\YasmineMnb\Desktop\june exp\200624_contin\1\Croped_3"
    i_frame = 0

    # add to video file
    first_img = cv2.imread(folder_path + "\\" + os.listdir(folder_path)[0])
    frame_height, frame_width, layers = first_img.shape
    save_video_name = folder_path + "\\"+ points_file.split("\\")[-1].strip(".txt") + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    video_out = cv2.VideoWriter(save_video_name, fourcc, 24.0, (frame_width, frame_height))

    loop_through_imgs(folder_path, points_file, i_frame, video_out)

if __name__ == "__main__":
    main()
#%%






