"""
dropfind.py
- Hrishee Shastri
"""

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import shutil 
import argparse
import tensorflow as tf 
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
import csv 
import time
from dir_watcher import DirWatcher
import sys # importyng sys in order to access scripts located in a different folder
import traceback
import datetime

path2scripts = 'research/' # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import

# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder




os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here

# specify which device you want to work on.
# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"]="0" # TODO: specify your computational device




# TODO: specify two pathes: to the pipeline.config file and to the folder with trained model.
path2config ='themodel/v5/pipeline.config'
path2model = 'themodel/v5/checkpoint'


# do not change anything in this cell
configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()

path2label_map = 'themodel/v5/label_map.pbtxt' # TODO: provide a path to the label map file
category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)


def detect_fn(image):
    """
    Detect objects in image.
    
    Args:
      image: (tf.tensor): 4D input image
      
    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """
    
    return np.array(Image.open(path))    



def nms(rects, thd=0.5):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []

    remove = [False] * len(rects)

    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


def inference_as_raw_output(image_path,
                            box_th = 0.25,
                            nms_th = 0.5):
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # checking how many detections we got
    num_detections = int(detections.pop('num_detections'))

    # filtering out detection in order to get only the one that are indeed detections
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # defining what we need from the resulting detection dict that we got from model output
    key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

    # filtering out detection dict in order to get only boxes, classes and scores
    detections = {key: value for key, value in detections.items() if key in key_of_interest}

    if box_th: # filtering detection if a confidence threshold for boxes was given as a parameter
        for key in key_of_interest:
            scores = detections['detection_scores']
            current_array = detections[key]
            filtered_current_array = current_array[scores > box_th]
            detections[key] = filtered_current_array

    if nms_th: # filtering rectangles if nms threshold was passed in as a parameter
        # creating a zip object that will contain model output info as
        output_info = list(zip(detections['detection_boxes'],
                               detections['detection_scores'],
                               detections['detection_classes']
                              )
                          )
        boxes, scores, classes = nms(output_info)
        
        detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
        detections['detection_scores'] = scores
        detections['detection_classes'] = classes
        

    return detections
         

def center_coords(image_path):
    """
    image_path : (str) path to image for inference

    Returns the coordinates (x,y) of the center of the drop with the 
            highest confidence
            If no drop identified, returns the center of the image by default
    """
    try:
        im = Image.open(image_path)
    except PIL.UnidentifiedImageError:
        time.sleep(5)
        im = Image.open(image_path)
        

    X_DIM, Y_DIM = im.size
    detections = inference_as_raw_output(image_path)
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    classes = detections['detection_classes']
    if len(boxes) == 0:
        return (X_DIM//2, Y_DIM//2)
    ind = scores.index(max(scores))
    box = boxes[ind] 
    # box is numpy array [y1, x1, y2, x2] where (x1,y1) are top left corner 
    # and (x2, y2) are bottom right corner, expressed as ratios of the total dimension

    # Convert to coordinate values
    y1 = box[0]*Y_DIM
    x1 = box[1]*X_DIM
    y2 = box[2]*Y_DIM
    x2 = box[3]*X_DIM
    result = (round(x1 + (x2 - x1)//2, 3), round(y1 + (y2 - y1)//2, 3))
    return result

def dropfind(dir_path, image):
    """
    image : (str) filename of image
    dir_path : (str) full path of directory where images are being populated
    
    creates and writes to a csv file "temp.csv" with two rows:
        image file name | (x,y) coordinates of inferred drop center
    """
    with open(dir_path + os.sep + "temp.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image, center_coords(dir_path + os.sep + image)])

def print_console_and_file(string, file, mute):
    if not mute:
        print(string)
    with open(file, 'a') as f:
        f.write(string + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to directory to be populated with images, and where results.csv will go")
    parser.add_argument("-n", "--num_ims", help="Number of images expected", type=int)
    parser.add_argument("-b", "--barcode", default=False, help="Barcode for the batch of images", type=str)
    parser.add_argument("-m", "--mute", default=True, help="Set True to suppress all output (print statements)", type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    dir_path = args.path 
    NUM_IMS = args.num_ims
    barcode = args.barcode
    mute = args.mute
    if NUM_IMS == 0:
        print("Imports successful")
    else:
        printcf = lambda string : print_console_and_file(string, file = "dropfind_log.txt", mute = mute)

        LOG_LIMIT = 500
        log_file = open("dropfind_log.txt", "a")
        num_lines = sum(1 for line in open("dropfind_log.txt"))
        if num_lines > LOG_LIMIT:
            log_file = open("dropfind_log.txt", "w")
            printcf("dropfind_log.txt contains over " + str(LOG_LIMIT) + "lines. Clearing...")
        else:
            log_file = open("dropfind_log.txt", "a")
        
        log_file.write("Monitoring directory: " + dir_path + "\n")
        log_file.write("Number of images expected: " + str(NUM_IMS) + "\n")
        log_file.write("Barcode: " + barcode + "\n")

        log_file.close()

        
        try:
            dirwait_timeout = 60
            x = 0
            dirwait_interval = 5
            DIR_FOUND_FLAG = True
            while DIR_FOUND_FLAG and not os.path.isdir(dir_path):
                printcf("Waiting for directory " + dir_path + " to be created...")
                time.sleep(dirwait_interval)
                x += dirwait_interval
                if x >= dirwait_timeout:
                    printcf("Timeout. Directory " + dir_path + " not found after " + str(dirwait_timeout) + " seconds. Quitting..." )
                    DIR_FOUND_FLAG = False

            if DIR_FOUND_FLAG:
                Watcher = DirWatcher(dir_path)
                INTERVAL = 1 # interval (in seconds) between refreshing directory for new images

                printcf("Directory " + dir_path + " found")
                printcf("Ready for inference...")
                n = 0
                while Watcher.count() < NUM_IMS:
                    files = Watcher.refresh_dir()
                    if Watcher.is_stop():
                        printcf("stop.txt detected. Quitting prematurely...")
                        break
                    for f in files:
                        n += 1
                        CURR_FILE = f
                        dropfind(dir_path, f)
                        printcf(str(n) + " Done - " + f)
                    time.sleep(INTERVAL)

                CURR_FILE = "Had finished inference on all images"

                new_fname = dir_path + os.sep + barcode + ".csv"
                os.rename(dir_path + os.sep + "temp.csv", new_fname)
                printcf("Renamed temp.csv to " + barcode + ".csv")

                with open(dir_path + os.sep + "exit.txt", 'w') as f:
                    f.write("exit acknowledgement")
                printcf("Deposited exit.txt acknowledgement to " + dir_path)

                Watcher.refresh_dir()
                assert(Watcher.is_exit())
                printcf("Quitting...")

            with open("dropfind_log.txt", "a") as log_file:
                log_file.write("---------------------------" +"\n")
        except:
            error_log = "dropfind_errors.txt"
            process_log = "dropfind_log.txt"
            with open(process_log, "a") as logfile:
                traceback.print_exc(file=logfile)
                logfile.write("----------------------------" + "\n")
            with open(error_log, "a") as logfile:
                logfile.write("Monitored Directory: " + dir_path + "\n")
                logfile.write("Date and Time: " + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
                try:
                    CURR_FILE = CURR_FILE
                except NameError:
                    CURR_FILE = "Had not started inference on images yet"

                logfile.write("Error on filename: " + CURR_FILE + "\n") 
                traceback.print_exc(file=logfile)
                logfile.write("----------------------------" + "\n")
            raise 
        

