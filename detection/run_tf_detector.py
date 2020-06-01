######
#
# run_tf_detector.py
#
# Functions to load a TensorFlow detection model, run inference,
# and render bounding boxes on images.
#
# See the "test driver" cell for example invocation.
#
######

#%% Constants, imports, environment

import tensorflow as tf
import numpy as np
import humanfriendly
import time
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os


#%% Core detection functions

def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    print('Creating Graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('...done')

    # MegaDetector was trained with batch size of 1, and the resizing function is a part
    # of the inference graph
    BATCH_SIZE = 1

    # An enumeration of failure reasons
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85  # to render bounding boxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '4': 'vehicle'  # will be available in megadetector v4
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads the model at model_path and start a tf.Session with this graph. The necessary
        input and output tensor handles are obtained also."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(np_array):
        """ Two effects: convert the numpy floats to Python floats, and also change the coordinates from
        [y1, x1, y2, x2] to [x1, y1, width_box, height_box] (in relative coordinates still).

        Args:
            np_array: array of predicted bounding box coordinates from the TF detector

        Returns: array of predicted bounding box coordinates as Python floats and in [x1, y1, width_box, height_box]

        """
        # change from [y1, x1, y2, x2] to [x1, y1, width_box, height_box]
        width_box = np_array[3] - np_array[1]
        height_box = np_array[2] - np_array[0]

        new = [np_array[1], np_array[0], width_box, height_box]  # cannot be a numpy array; needs to be a list

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coords(array):
        """ From [x1, y1, width_box, height_box] to [y1, x1, y2, x2]"""
        x2 = array[0] + array[2]
        y2 = array[1] + array[3]
        return [array[0], array[1], x2, y2]

    @staticmethod
    def __load_model(model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.

        Args:
            model_path: .pb file of the model.

        Returns: the loaded graph.

        """
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id,
                                      detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """Apply the detector to an image.

        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the `file` field of the output object
            detection_threshold: confidence above which to include the detection proposal

        Returns:
        A dict with the following fields, see https://github.com/microsoft/CameraTraps/tree/siyu/inference_refactor/api/batch_processing#batch-processing-api-output-format
            - image_id (always present)
            - max_detection_conf
            - detections, which is a list of detection objects containing `category`, `conf` and `bbox`
            - failure
        """
        result = {
            'file': image_id
        }
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'category': str(int(c)),  # use string type for the numerical class label, not int
                        'conf': truncate_float(float(s),  # cast to float for json serialization
                                               precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = truncate_float(float(max_detection_conf),
                                                          precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result


def generate_detections(detection_graph,images):
    """
    boxes,scores,classes,images = generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] can be a list of numpy arrays or a list of filenames.  Non-list inputs will be
    wrapped into a list.

    Boxes are returned in relative coordinates as (top, left, bottom, right); x,y origin is the upper-left.
    """

    if not isinstance(images,list):
        images = [images]
    else:
        images = images.copy()

    # Load images if they're not already numpy arrays
    for iImage,image in enumerate(images):
        if isinstance(image,str):
            print('Loading image {}'.format(image))
            # image = Image.open(image).convert("RGBA"); image = np.array(image)
            image = mpimg.imread(image)
            images[iImage] = image
        else:
            assert isinstance(image,np.ndarray)

    boxes = []
    scores = []
    classes = []
    
    nImages = len(images)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for iImage,imageNP in enumerate(images):                
                print('Processing image {} of {}'.format(iImage,nImages))
                imageNP_expanded = np.expand_dims(imageNP, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                score = detection_graph.get_tensor_by_name('detection_scores:0')
                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                
                # Actual detection
                (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: imageNP_expanded})

                boxes.append(box)
                scores.append(score)
                classes.append(clss)
                
    boxes = np.squeeze(np.array(boxes))
    scores = np.squeeze(np.array(scores))
    classes = np.squeeze(np.array(classes)).astype(int)

    return boxes,scores,classes,images


#%% Rendering functions

def render_bounding_boxes(boxes,scores,classes,inputFileNames,outputFileNames=[],confidenceThreshold=0.9):
    """
    Render bounding boxes on the image files specified in [inputFileNames].  [boxes] and [scores] should be in the format
    returned by generate_detections.
    """

    nImages = len(inputFileNames)
    iImage = 0

    for iImage in range(0,nImages):

        inputFileName = inputFileNames[iImage]

        if iImage >= len(outputFileNames):
            outputFileName = ''
        else:
            outputFileName = outputFileNames[iImage]

        if len(outputFileName) == 0:
            name, ext = os.path.splitext(inputFileName)
            outputFileName = "{}.{}{}".format(name,'_detections',ext)

        image = mpimg.imread(inputFileName)
        iBox = 0; box = boxes[iImage][iBox]
        dpi = 100
        s = image.shape; imageHeight = s[0]; imageWidth = s[1]
        figsize = imageWidth / float(dpi), imageHeight / float(dpi)

        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1])
        
        # Display the image
        ax.imshow(image)
        ax.set_axis_off()
    
        # plt.show()
        for iBox,box in enumerate(boxes[iImage]):

            score = scores[iImage][iBox]
            if score < confidenceThreshold:
                continue

            # top, left, bottom, right 
            #
            # x,y origin is the upper-left
            topRel = box[0]
            leftRel = box[1]
            bottomRel = box[2]
            rightRel = box[3]
            
            x = leftRel * imageWidth
            y = topRel * imageHeight
            w = (rightRel-leftRel) * imageWidth
            h = (bottomRel-topRel) * imageHeight
            
            # Location is the bottom-left of the rect
            #
            # Origin is the upper-left
            iLeft = x
            iBottom = y
            rect = patches.Rectangle((iLeft,iBottom),w,h,linewidth=2,edgecolor='r',facecolor='none')
            
            # Add the patch to the Axes
            ax.add_patch(rect)        

        # ...for each box

        # This is magic goop that removes whitespace around image plots (sort of)        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.axis('tight')
        ax.set(xlim=[0,imageWidth],ylim=[imageHeight,0],aspect=1)
        plt.axis('off')                

        # plt.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)
        plt.savefig(outputFileName, dpi=dpi, transparent=True)
        # os.startfile(outputFileName)

    # ...for each image

# ...def render_bounding_boxes


#%% Test driver

MODEL_FILE = r'd:\temp\models\frozen_inference_graph.pb'
TARGET_IMAGES = [r'd:\temp\test_images\d16558s6i1.jpg',r'D:\temp\test_images\d16558s1i6.jpg']

# Load and run detector on target images
detection_graph = load_model(MODEL_FILE)

startTime = time.time()
boxes,scores,classes,images = generate_detections(detection_graph,TARGET_IMAGES)
elapsed = time.time() - startTime
print("Done running detector on {} files in {}".format(len(images),humanfriendly.format_timespan(elapsed)))

assert len(boxes) == len(TARGET_IMAGES)

inputFileNames = TARGET_IMAGES
outputFileNames=[]
confidenceThreshold=0.9

plt.ioff()

render_bounding_boxes(boxes, scores, classes, TARGET_IMAGES)
