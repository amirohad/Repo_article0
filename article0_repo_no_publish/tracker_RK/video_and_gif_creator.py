"""
using opencv V4.0.0 and python 3.6

lets you create video files (.avi and .mp4 are included, anything else you would
need to read up on... )
"""

import cv2
import os
import numpy as np

## These are python file that need to be included in the same folder
import img_processing
import progress_bar

## for gif making
from PIL import Image

def create_video(input_folder_path, outvid_path, fps):
    r"""
    create video from images in input folder
    input_folder_path should be full path:
        'C:\Users\...\my_images'
    outvid_path should be with the file type you chose (writen here as .avi)
        'C:\Users\...\my_video.avi'
    fps - frames per second (needs to be a float type!)
    """
    imgs_lst = os.listdir(input_folder_path)

## There might be a limit to what your video player can show so we resize
# ================= Resize Option 1: ===========================================
#     ## Get size from first img
#     image0 = input_folder_path +"\\"+ imgs_lst[0]
#     img0 = cv2.imread(image0)
#
#     ## We take only half the size
#     size = (int(img0.shape[1]/2), int(img0.shape[0]/2))
#     size = ((img0.shape[1]), (img0.shape[0]))
#     size = (640,480)
# ================ Resize Option 2: ===========================================

    ## Choose your resolution
    size = (1980,1080)
    size = (1080,720)

    ## set params for the video output
    is_color = True
    fourcc = cv2.VideoWriter_fourcc(*"XVID")   # .avi
#    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # .mp4
    vid = cv2.VideoWriter(outvid_path, fourcc, fps, size, is_color)

    try:
        for i in range(0, int(len(imgs_lst))):
            image_file = input_folder_path +"\\"+ imgs_lst[i]
            img = cv2.imread(image_file)

			## catch common errors, this would happen if the img isn't loaded properly
            if type(img) != np.ndarray:
                print("Error reading img file")
                break

# ========== if anything should be done with img enter code here ==============
#
#            flipHorizontal = cv2.flip(originalImage, 1)
            img = cv2.resize(img, size)
#            img = cv2.fastNlMeansDenoisingColored(img, None, 15, 10, 7, 21)
#
			## add time stamp
            creation_time = img_processing.get_time(image_file)
            img_processing.text_on_img(img, creation_time)
# =============================================================================

            ## show img while processing
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", img)
            cv2.waitKey(1)

            ## write to video file
            vid.write(img)

            ## for the progress bar:
            perc = i/len(imgs_lst)
            progress_bar.update_progress_bar(perc)

        ## update progress bar we finished:
        progress_bar.update_progress_bar(1)

    except KeyboardInterrupt:
        print("cought: Keyboard Interrupt")

    ## finish up nicely and release the file
    finally:
        vid.release()
        cv2.destroyAllWindows()

def create_gif(input_folder_path, output_path):
    r"""
    creates a gif out of a folder full with images
    outvid_path should have .gif ending
        r'C:\Users\...\my_video.gif'
    """
    ## Load list of images
    imgs_lst = os.listdir(input_folder_path)

    frames = []
    for i in range(0, int(len(imgs_lst))):
        ## load current image
        image_file = input_folder_path +"\\"+ imgs_lst[i]
        img = cv2.imread(image_file)

        ## resize
        size = (640,480)
        img = cv2.resize(img, size)

        ## show current image
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(1)

        ## convert to PIL Image format and add to frames list
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        frames.append(im_pil)

        ## update progress bar
        perc = i/len(imgs_lst)
        progress_bar.update_progress_bar(perc, "time")

    print("now creating gif this may take a moment...")
    frames[0].save(output_path, format='GIF',
                      append_images=frames[1:500], save_all=True, duration=40, loop=0)
    print("Done! :)")


def main():
    input_folder_path = r"C:\Users\YasmineMnb\Desktop\SynologyDrive\proper_experiments\200806_contin\1_L\Croped_1"
    outvid_path = r"C:\Users\YasmineMnb\Desktop\SynologyDrive\proper_experiments\200806_contin\1_L\Croped_1.avi"

    create_video(input_folder_path, outvid_path, 16.0)
    create_gif(input_folder_path,outvid_path.replace(".avi", ".gif"))

if __name__ == '__main__':
    main()