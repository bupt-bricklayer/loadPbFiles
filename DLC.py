import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv

PB_PATH = 'E:/output/pb/DLC/DLC_Model_R0402s.pb'


def preprocess(resized_inputs):
    return (2.0 / 255.0) * resized_inputs - 1.0
# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 1)).astype(np.uint8)

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PB_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        writer = tf.summary.FileWriter('E:/output/tensorboard', sess.graph)
        image_tensor = detection_graph.get_tensor_by_name('input_1:0')
        res_tensor = detection_graph.get_tensor_by_name('pred/convolution:0')
        image = Image.open("E:/output/pic_input/1.tiff")

        image = image.resize((512, 512))
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, 512, 512, 1]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # image_np_expanded = preprocess(image_np_expanded)
        # Actual detection.
        (res) = sess.run(
            [res_tensor],
            feed_dict={image_tensor: image_np_expanded})
        pic = np.array(res)
        pic = pic.reshape((512,512))
        cv.imwrite("E:/output/pic_output/1.tiff", pic)
        print(pic)
