import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tff
from glob import glob
import sys
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


class TLClassifier(object):
    def __init__(self, model_path):
        self.traffic_light_box = None
        self.classified_index = 0
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            self.od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

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
            feed_dict={self.image_tensor: image_np_expanded}
        )

        self.sq_scores = np.squeeze(scores)
        self.sq_classes = np.squeeze(classes)
        self.sq_boxes = np.squeeze(boxes)

        sel_id = np.logical_and(self.sq_classes == target_class, self.sq_scores > score_threshold)
        return self.sq_boxes[sel_id]

    def draw_rectangles_on_image(self,image_np, boxes):
        """Draw rectangles around detected traffic lights."""
        image_pil = Image.fromarray(np.uint8(image_np))  # Convert to PIL image
        draw = ImageDraw.Draw(image_pil)
        im_height, im_width, _ = image_np.shape

        for box in boxes:
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = (
                xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height
            )

            # Draw rectangle
            draw.rectangle(
                [(left, top), (right, bottom)],
                outline="red",
                width=5
            )

        return image_pil


    def process_image_from_input_folder(self, image_folder_evaluation):
        for filename in os.listdir(image_folder_evaluation):
            file_to_open = os.path.join(image_folder_evaluation, filename)

            if os.path.isfile(file_to_open):
                im = Image.open(file_to_open)
                image_np = np.asarray(im)
                boxes = self.detect_multi_object(image_np, score_threshold=0.2)

                if len(boxes) > 0:
                    print(f"Traffic light detected in: {file_to_open}")
                    image_with_rectangles = self.draw_rectangles_on_image(image_np, boxes)
                    plt.imshow(image_with_rectangles)
                    plt.show()
                else:
                    print(f"No traffic light detected in: {file_to_open}")

def main():
    PATH_TO_CKPT = 'img/trainDataset/frozen_inference_graph.pb'
    image_folder_evaluation = "img/evaluation"
    tlc = TLClassifier(PATH_TO_CKPT)
    tlc.process_image_from_input_folder(image_folder_evaluation)

if __name__ == "__main__":
    main()
#         / \
#  -->>>>  | start doing at here
