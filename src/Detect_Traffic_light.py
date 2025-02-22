import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
from PIL import Image
import pillow_heif
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from skimage.color import rgb2gray,rgb2hsv
import warnings
import tensorflow as tf

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress deprecation warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

#################################################################################################################


class TLClassifier(object):
    def __init__(self, model_path):
        #######################################################
        self.traffic_light_box = None
        self.classified_index = 0

        ######################################################
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

        ####################################################
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.TF_Session = tf.compat.v1.Session(graph=self.detection_graph)
        


    def detect_multi_object(self, image_np, score_threshold, target_class=10):
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = self.TF_Session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        #######################################################
        self.sq_scores = np.squeeze(scores)
        self.sq_classes = np.squeeze(classes)
        self.sq_boxes = np.squeeze(boxes)
        ##################################################        
        sel_id = np.logical_and(self.sq_classes == target_class, self.sq_scores > score_threshold)
        return self.sq_boxes[sel_id]





    # Crops the region of interest (ROI) from the input image based on the bounding box.
    # image_np: Input image in NumPy array format.
    # sel_box: Selected bounding box coordinates.


    def crop_roi_image(self, image_np, sel_box):
        im_height, im_width, _ = image_np.shape
        left, right, top, bottom = (sel_box[1] * im_width, sel_box[3] * im_width,
                                     sel_box[0] * im_height, sel_box[2] * im_height)
        return image_np[int(top):int(bottom), int(left):int(right), :]


#####################################################################################################################################

def process_image_from_input_folder(tlc, PATH_TO_CKPT, image_folder_evaluation ):

    for filename in os.listdir(image_folder_evaluation):
        file_to_open = os.path.join(image_folder_evaluation, filename)
        if os.path.isfile(file_to_open):
            im = Image.open(file_to_open)
            image_np = np.asarray(im)
            boxes = tlc.detect_multi_object(image_np, score_threshold=0.2)

            if len(boxes) > 0:
                cropped_image = tlc.crop_roi_image(image_np, boxes[0])
                plt.imshow(cropped_image)
                Image_found = Image.fromarray(np.uint8(cropped_image))

                print(f"Traffic light detected in: {file_to_open}")
            else:
                print(f"No traffic light in: {file_to_open}")

#######################################################################################################

def main():
    PATH_TO_CKPT =  'img/trainDataset/frozen_inference_graph.pb'
    image_folder_evaluation = "img/evaluation"
    tlc = TLClassifier(PATH_TO_CKPT)
    process_image_from_input_folder(tlc, PATH_TO_CKPT, image_folder_evaluation)

if __name__ == "__main__":
    main()