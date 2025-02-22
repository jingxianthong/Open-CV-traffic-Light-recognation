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



PATH_TO_CKPT =  'frozen_inference_graph.pb'
def load_graph():

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

def select_boxes(boxes, classes, scores, score_threshold=0, target_class=10):
    """

    :param boxes:
    :param classes:
    :param scores:
    :param target_class: default traffic light id in COCO dataset is 10
    :return:
    """

    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)

    return sq_boxes[sel_id]


class TLClassifier(object):
    def __init__(self):

        self.detection_graph = load_graph()
        self.extract_graph_components()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # run the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_multi_object(dummy_image,0.1)
        self.traffic_light_box = None
        self.classified_index = 0

    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def detect_multi_object(self, image_np, score_threshold):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :return:
        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        sel_boxes = select_boxes(boxes=boxes, classes=classes, scores=scores,
                                 score_threshold=score_threshold, target_class=10)

        return sel_boxes


#########################################################################################################
test_file = "streetimage2.jpg"

from PIL import Image
im = Image.open(test_file)
image_np = np.asarray(im)

plt.imshow(image_np)

tlc=TLClassifier()

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:02.713377Z","iopub.execute_input":"2022-05-14T12:24:02.713621Z","iopub.status.idle":"2022-05-14T12:24:02.759634Z","shell.execute_reply.started":"2022-05-14T12:24:02.713588Z","shell.execute_reply":"2022-05-14T12:24:02.758835Z"}}
def crop_roi_image(image_np, sel_box):
    im_height, im_width, _ = image_np.shape
    (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                  sel_box[0] * im_height, sel_box[2] * im_height)
    cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
    return cropped_image


boxes=tlc.detect_multi_object(image_np,score_threshold=0.2)

if len(boxes)>0:
    cropped_image=crop_roi_image(image_np,boxes[0])
    plt.imshow(cropped_image)
    immm = Image.fromarray(cropped_image)
    outputfile_name = test_file + ".jpg"
    immm.save(outputfile_name)
    print(f"Traffice light for: {test_file}")
else:
    print("No traffic light")

if 'cropped_image' not in locals():
    exit()

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:03.055098Z","iopub.execute_input":"2022-05-14T12:24:03.055404Z","iopub.status.idle":"2022-05-14T12:24:04.056604Z","shell.execute_reply.started":"2022-05-14T12:24:03.055351Z","shell.execute_reply":"2022-05-14T12:24:04.055721Z"}}
from skimage.color import rgb2gray,rgb2hsv
hsv_test_image=rgb2hsv(cropped_image)

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:04.058186Z","iopub.execute_input":"2022-05-14T12:24:04.058499Z","iopub.status.idle":"2022-05-14T12:24:04.278325Z","shell.execute_reply.started":"2022-05-14T12:24:04.058449Z","shell.execute_reply":"2022-05-14T12:24:04.277493Z"}}
plt.hist(hsv_test_image[:,:,1])
plt.xlabel("hsv_test_image")
plt.show()
###exit()

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:04.279566Z","iopub.execute_input":"2022-05-14T12:24:04.279805Z","iopub.status.idle":"2022-05-14T12:24:04.582494Z","shell.execute_reply.started":"2022-05-14T12:24:04.279764Z","shell.execute_reply":"2022-05-14T12:24:04.581349Z"}}
s_val_1d=hsv_test_image[:,:,1].ravel()
plt.hist(s_val_1d)
plt.xlabel("s_val_1d")
plt.show()

#         / \
#  -->>>>  |start doing at here

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:04.583842Z","iopub.execute_input":"2022-05-14T12:24:04.584132Z","iopub.status.idle":"2022-05-14T12:24:04.913552Z","shell.execute_reply.started":"2022-05-14T12:24:04.584067Z","shell.execute_reply":"2022-05-14T12:24:04.912566Z"}}
saturation_threshold=0.8
idx = hsv_test_image[:, :, 1] <=saturation_threshold
sat_mask = np.ones_like(hsv_test_image[:, :, 1])
sat_mask[idx] = 0
fig,ax=plt.subplots(nrows=1,ncols=2)
ax[0].imshow(hsv_test_image[:,:,1])
ax[1].imshow(sat_mask)

# %% [markdown]
# We perform similar operation for Value channle. This time I used my written function:

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:04.914946Z","iopub.execute_input":"2022-05-14T12:24:04.915269Z","iopub.status.idle":"2022-05-14T12:24:04.967452Z","shell.execute_reply.started":"2022-05-14T12:24:04.915199Z","shell.execute_reply":"2022-05-14T12:24:04.966478Z"}}
def high_value_region_mask(hsv_image, v_thres=0.6):
    if  np.issubdtype(hsv_image.dtype, np.integer):
        idx = (hsv_image[:, :, 2].astype(np.float) / 255.0) < v_thres
    else:
        idx = (hsv_image[:, :, 2].astype(np.float)) < v_thres
    mask = np.ones_like(hsv_image[:, :, 2])
    mask[idx] = 0
    return mask

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:04.96856Z","iopub.execute_input":"2022-05-14T12:24:04.968861Z","iopub.status.idle":"2022-05-14T12:24:05.174479Z","shell.execute_reply.started":"2022-05-14T12:24:04.968804Z","shell.execute_reply":"2022-05-14T12:24:05.173603Z"}}
v_thres_val=0.9
val_mask=high_value_region_mask(hsv_test_image,v_thres=v_thres_val)
plt.imshow(val_mask)

# %% [markdown]
# By performing these two masks, we reach the following mask for selecting the region to calculate the average hue values

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:05.176161Z","iopub.execute_input":"2022-05-14T12:24:05.176487Z","iopub.status.idle":"2022-05-14T12:24:05.228966Z","shell.execute_reply.started":"2022-05-14T12:24:05.176436Z","shell.execute_reply":"2022-05-14T12:24:05.227876Z"}}
def get_masked_hue_image(hsv_test_image):

    s_thres_val = channel_percentile(hsv_test_image[:, :, 1], percentile=70)
    v_thres_val = channel_percentile(hsv_test_image[:, :, 2], percentile=70)
    val_mask = high_value_region_mask(hsv_test_image, v_thres=v_thres_val)
    sat_mask = high_saturation_region_mask(hsv_test_image, s_thres=s_thres_val)
    masked_hue_image = hsv_test_image[:, :, 0]
    return masked_hue_image

# %% [markdown]
# ## Filter out the region of interest

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:05.230553Z","iopub.execute_input":"2022-05-14T12:24:05.23093Z","iopub.status.idle":"2022-05-14T12:24:05.285809Z","shell.execute_reply.started":"2022-05-14T12:24:05.230866Z","shell.execute_reply":"2022-05-14T12:24:05.28489Z"}}
def high_saturation_region_mask(hsv_image, s_thres=0.6):
    if hsv_image.dtype == np.int:
        idx = (hsv_image[:, :, 1].astype(np.float) / 255.0) < s_thres
    else:
        idx = (hsv_image[:, :, 1].astype(np.float)) < s_thres
    mask = np.ones_like(hsv_image[:, :, 1])
    mask[idx] = 0
    return mask


def channel_percentile(single_chan_image, percentile):
    sq_image = np.squeeze(single_chan_image)
    assert len(sq_image.shape) < 3

    thres_value = np.percentile(sq_image.ravel(), percentile)

    return float(thres_value) / 255.0

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:05.287036Z","iopub.execute_input":"2022-05-14T12:24:05.287416Z","iopub.status.idle":"2022-05-14T12:24:05.6333Z","shell.execute_reply.started":"2022-05-14T12:24:05.287348Z","shell.execute_reply":"2022-05-14T12:24:05.632395Z"}}
hue_image=hsv_test_image[:,:,0]
fig,ax=plt.subplots(nrows=1,ncols=2)
ax[0].imshow(hue_image,cmap='hsv')
ax[0].set_title("hue image")
ax[1].imshow(np.logical_and(sat_mask,val_mask))
ax[1].set_title("mask to be applied")

#TODO!!!!! redraw this image. Use hue color map, make zero values to "black

# %% [markdown]
# Note that the following statement is not equivalent to
# ```
# masked_hue_1d= (maksed_hue_image*np.logical_and(val_mask,sat_mask)).ravel()
# ```
# Because zero in hue channel means red, we cannot just set unused pixels to zero.

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:05.63491Z","iopub.execute_input":"2022-05-14T12:24:05.635177Z","iopub.status.idle":"2022-05-14T12:24:05.923387Z","shell.execute_reply.started":"2022-05-14T12:24:05.635129Z","shell.execute_reply":"2022-05-14T12:24:05.922379Z"}}
hue_1d=hue_image[np.logical_and(val_mask, sat_mask)].ravel()

plt.hist(hue_1d,bins=50)
plt.xlabel("hue")
plt.ylabel("occurences")
plt.show()

# %% [markdown]
# The hue values in this region is near 0.5, which is green.

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:05.924755Z","iopub.execute_input":"2022-05-14T12:24:05.925047Z","iopub.status.idle":"2022-05-14T12:24:05.980883Z","shell.execute_reply.started":"2022-05-14T12:24:05.924994Z","shell.execute_reply":"2022-05-14T12:24:05.979876Z"}}
def get_masked_hue_values(rgb_image):
    """
    Get the pixels in the RGB image that has high saturation (S) and value (V) in HSV chanels

    :param rgb_image: image (height, width, channel)
    :return: a 1-d array
    """

    hsv_test_image = rgb2hsv(rgb_image)
    s_thres_val = channel_percentile(hsv_test_image[:, :, 1], percentile=30)
    v_thres_val = channel_percentile(hsv_test_image[:, :, 2], percentile=70)
    val_mask = high_value_region_mask(hsv_test_image, v_thres=v_thres_val)
    sat_mask = high_saturation_region_mask(hsv_test_image, s_thres=s_thres_val)
    masked_hue_image = hsv_test_image[:, :, 0] * 180
    # Note that the following statement is not equivalent to
    # masked_hue_1d= (maksed_hue_image*np.logical_and(val_mask,sat_mask)).ravel()
    # Because zero in hue channel means red, we cannot just set unused pixels to zero.
    masked_hue_1d = masked_hue_image[np.logical_and(val_mask, sat_mask)].ravel()

    return masked_hue_1d

def convert_to_hue_angle(hue_array):
    """
    Convert the hue values from [0,179] to radian degrees [-pi, pi]

    :param hue_array: array-like, the hue values in degree [0,179]
    :return: the angles of hue values in radians [-pi, pi]
    """

    hue_cos = np.cos(hue_array * np.pi / 90)
    hue_sine = np.sin(hue_array * np.pi / 90)

    hue_angle = np.arctan2(hue_sine, hue_cos)

    return hue_angle



# %% [markdown]
# The following codes detects the color by hue values of an image.

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:05.982348Z","iopub.execute_input":"2022-05-14T12:24:05.982594Z","iopub.status.idle":"2022-05-14T12:24:06.041248Z","shell.execute_reply.started":"2022-05-14T12:24:05.982554Z","shell.execute_reply":"2022-05-14T12:24:06.040321Z"}}
def get_rgy_color_mask(hue_value, from_01=False):
    """
    return a tuple of np.ndarray that sets the pixels with red, green and yellow matrices to be true

    :param hue_value:
    :param from_01: True if the hue values is scaled from 0-1 (scikit-image), otherwise is -pi to pi
    :return:
    """

    if from_01:
        n_hue_value = conver_to_hue_angle_from_01(hue_value)
    else:
        n_hue_value = hue_value

    red_index = np.logical_and(n_hue_value < (0.125 * np.pi), n_hue_value > (-0.125 * np.pi))

    green_index = np.logical_and(n_hue_value > (0.66 * np.pi), n_hue_value < np.pi)

    yellow_index = np.logical_and(n_hue_value > (0.25 * np.pi), n_hue_value < (5.0 / 12.0 * np.pi))

    return red_index, green_index, yellow_index


def classify_color_by_range(hue_value):
    """
    Determine the color (red, yellow or green) in a hue value array

    :param hue_value: hue_value is radians
    :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
    """

    red_index, green_index, yellow_index = get_rgy_color_mask(hue_value)

    color_counts = np.array([np.sum(red_index) / len(hue_value),
                             np.sum(yellow_index) / len(hue_value),
                             np.sum(green_index) / len(hue_value)])

    color_text = ['red', 'yellow', 'green', '_', 'unknown']

    min_index = np.argmax(color_counts)

    return min_index, color_text[min_index]

def classify_color_cropped_image(rgb_image):
    """
    Full pipeline of classifying the traffic light color from the traffic light image

    :param rgb_image: the RGB image array (height,width, RGB channel)
    :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
    """

    hue_1d_deg = get_masked_hue_values(rgb_image)

    if len(hue_1d_deg) == 0:
        return 4, 'unknown'

    hue_1d_rad = convert_to_hue_angle(hue_1d_deg)

    return classify_color_by_range(hue_1d_rad)

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:06.042757Z","iopub.execute_input":"2022-05-14T12:24:06.04305Z","iopub.status.idle":"2022-05-14T12:24:06.098222Z","shell.execute_reply.started":"2022-05-14T12:24:06.043Z","shell.execute_reply":"2022-05-14T12:24:06.097083Z"}}
classify_color_cropped_image(cropped_image)

# %% [markdown]
# # Put all things together to classify an image

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:06.101174Z","iopub.execute_input":"2022-05-14T12:24:06.101836Z","iopub.status.idle":"2022-05-14T12:24:06.245635Z","shell.execute_reply.started":"2022-05-14T12:24:06.101535Z","shell.execute_reply":"2022-05-14T12:24:06.244766Z"}}
boxes=tlc.detect_multi_object(image_np,score_threshold=0.1)

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:06.24677Z","iopub.execute_input":"2022-05-14T12:24:06.247065Z","iopub.status.idle":"2022-05-14T12:24:06.298622Z","shell.execute_reply.started":"2022-05-14T12:24:06.247029Z","shell.execute_reply":"2022-05-14T12:24:06.297773Z"}}
def classify_all_boxes_in_image(image_np, boxes):
    result_index_array = np.zeros(boxes.shape[0], dtype=np.int)
    for i, box in enumerate(boxes):
        cropped_image = crop_roi_image(image_np, box)
        result_color_index, _ = classify_color_cropped_image(cropped_image)
        result_index_array[i] = result_color_index

    return result_index_array

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:06.299944Z","iopub.execute_input":"2022-05-14T12:24:06.300491Z","iopub.status.idle":"2022-05-14T12:24:06.355728Z","shell.execute_reply.started":"2022-05-14T12:24:06.300424Z","shell.execute_reply":"2022-05-14T12:24:06.354636Z"}}
results_index=classify_all_boxes_in_image(image_np,boxes)



import numpy as np
import matplotlib.pyplot as plt

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

# color maps for drawing the resulting identified traffic light color on the image.
COLOR_MAP = ['#f44141',  # red,
             '#f1f441',  # yellow
             '#a7f442',  # green
             '#ffffff',  # white
             '#ffffff',  # white
             ]


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
    """A
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).

    Args:
      image: a numpy array object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                 display_str_list_list)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
    """Draws bounding boxes on image.

    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                   boxes[i, 3], color, thickness, display_str_list)

def draw_result_on_image(image_np, box, tl_result):
    """
    Draw color box based on the calssifed traffic light results
    :param image_np: the image to draw on
    :param box: the normalized box index
    :param tl_result: the index returned by classify_color_v2()
    :return:
    """
    ymin, xmin, ymax, xmax = box
    color = COLOR_MAP[tl_result]
    draw_bounding_box_on_image_array(image_np, ymin, xmin, ymax, xmax, color=color)
    return image_np

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:06.441107Z","iopub.execute_input":"2022-05-14T12:24:06.441628Z","iopub.status.idle":"2022-05-14T12:24:06.490673Z","shell.execute_reply.started":"2022-05-14T12:24:06.441584Z","shell.execute_reply":"2022-05-14T12:24:06.489805Z"}}
def draw_results_on_image(image_np, boxes, tl_results_array):
    for i, box in enumerate(boxes):
        draw_result_on_image(image_np, box, tl_results_array[i])

# %% [code] {"execution":{"iopub.status.busy":"2022-05-14T12:24:06.491973Z","iopub.execute_input":"2022-05-14T12:24:06.492769Z","iopub.status.idle":"2022-05-14T12:24:06.863311Z","shell.execute_reply.started":"2022-05-14T12:24:06.492715Z","shell.execute_reply":"2022-05-14T12:24:06.862439Z"}}
n_image_np=np.copy(image_np)
draw_results_on_image(n_image_np, boxes, results_index)
plt.hist(n_image_np)
plt.show()

