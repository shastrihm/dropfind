"""
dropfind.py
- Hrishee Shastri
- September 2021
input : no. of ims and path to directory
starting scan in this directory
    - monitoring loop of that directory
        - if new image found, infer and write to csv file
    - output: csv file into same directory
        image name, coordinates of center
"""


import os # importing OS in order to make GPU visible
import shutil 
import argparse
import tensorflow as tf 
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv 
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys # importyng sys in order to access scripts located in a different folder

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
path2config ='themodel/pipeline.config'
path2model = 'themodel/checkpoint'


# do not change anything in this cell
configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()

path2label_map = 'themodel/label_map.pbtxt' # TODO: provide a path to the label map file
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

    return (x1 + (x2 - x1)//2, y1 + (y2 - y1)//2)


def dropfind(dir_path, image):
    """
    image : (str) filename of image
    dir_path : (str) full path of directory where images are being populated
    
    creates and writes to a csv file "results.csv" with two rows:
        image file name | (x,y) coordinates of inferred drop center
    """
    with open(dir_path + os.sep + "results.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image, center_coords(dir_path + os.sep + image)])




COUNT = 0

class Watcher:
    def __init__(self, directory, num_ims):
        self.observer = Observer()
        self.DIRECTORY_TO_WATCH = directory
        self.n = num_ims
        self.count = 0

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        
        if test_mode:
            print("TESTING INSTALLATION")
            source = "test_install\\"
            target = "test_install\\test\\"
            for fname in os.listdir(source):
                if fname[-4:] == ".jpg":
                    shutil.move(os.path.join(source, fname), target)
        try:
            while True:
                global COUNT 
                if COUNT == NUM_IMS:
                    sys.exit(0)
                time.sleep(5)
        except:
            self.observer.stop()
            print("Quitting...")
            if test_mode:
                print("CLEANING UP TESTBED")
                target = "test_install\\"
                source = "test_install\\test\\"
                for fname in os.listdir(source):
                    if fname[-4:] == ".jpg":
                        shutil.move(os.path.join(source, fname), target)

                # Validating output csv file
                assert "results.csv" in os.listdir(source), "results.csv not found"
                file = open(source + os.sep + "results.csv")
                reader = csv.reader(file)
                assert len(list(reader)) == NUM_IMS, "results.csv has incorrect no. of rows"
                file.close()
                os.remove(source + os.sep + "results.csv")

                print()
                print("Test passed. Installation Successful.")

        self.observer.join()


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            head, tail = os.path.split(event.src_path)
            if tail != "results.csv":
                global COUNT
                COUNT += 1
                dropfind(head, tail)
                # Take any action here when a file is first created.
                print("Done - %s." % tail)
                if COUNT == NUM_IMS:
                    print(str(COUNT) + " images done")
                    sys.exit(0)
                    

                    

           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to directory to be populated with images, and where results.csv will go")
    parser.add_argument("-n", "--num_ims", help="Number of images expected", type=int)
    parser.add_argument("-t", "--test", default=False, help="Set True if testing installation, Set False for regular use", type=bool)

    args = parser.parse_args()
    dir_path = args.path 
    NUM_IMS = args.num_ims
    test_mode = args.test
    print()
    print("Ready for inference...")
    w = Watcher(dir_path, NUM_IMS)
    w.run()


