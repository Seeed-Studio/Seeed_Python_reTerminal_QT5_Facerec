import cv2
import numpy as np
import skimage 
import skimage.transform
import json, base64
import time 
import scipy 

from tflite_runtime.interpreter import Interpreter

from anchors import ANCHOR

IMG_SHAPE = [96, 112]
OBJ_THRES = 0.7
NMS_THRES = 0.4
VARIANCE = [0.1, 0.2]

offset_x = 0
offset_y = -5

src = np.array([[32.82394272, 51.69630032+offset_y],
                [63.02725728, 51.50140016+offset_y],
                [48.0216,     71.73660032+offset_y],
                [35.61368544, 92.36549952],
                [60.625632,   92.20409968]])

def write_db(db, id, name, vector):
    for item in db:
        db[item]['vector'] = base64.b64encode(db[item]['vector']).decode('utf-8')

    vector = base64.b64encode(vector).decode('utf-8')
    entry = {"name": name, "vector": vector}
    db[id] = entry
    print(db)
    f = open('resources/database.db','w')
    entry = json.dumps(db)
    f.write(entry)
    f.close()

    return db

def read_db(db_path = 'resources/database.db'):
    try:
        f = open(db_path, 'r')
    except FileNotFoundError:
        clear_db(db_path)
        f = open(db_path, 'r')

    content = f.read()
    if content:
        db = json.loads(content)
    f.close()

    for item in db:
        db[item]['vector'] = np.frombuffer(base64.b64decode(db[item]['vector']), np.float32)

    return db

def clear_db(db_path = 'database.db'):

    f = open(db_path,'w')
    db = {}
    content = json.dumps(db)
    f.write(content)
    f.close()


def preprocess_for_fd(img):

    img = img.astype(np.float32)
    img = (img / 255) - 0.5
    img = np.expand_dims(img, 0)
    return img


def preprocess_for_fe(img):
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, 0)
    return img

def pred_boxes(box_output, score_output, ldmk_output):
    '''
    generate box information from output
    :param box_output: 3160*4
    :param score_output: 3160*2
    :param ldmk_output: 3160*10
    :return:
    '''

    # select boxes greater than threshold probability
    prob = scipy.special.softmax(score_output, axis=-1)
    pre_select_boxes_mask = prob[:, 1] > OBJ_THRES
    pre_select_boxes_index = np.where(pre_select_boxes_mask)[0]
    pre_select_anchor = ANCHOR[pre_select_boxes_index, :]

    # calculate coordinate
    box_cord = box_output[pre_select_boxes_index, :]
    box_cord = np.concatenate((
        pre_select_anchor[:, :2] + box_cord[:, :2] * VARIANCE[0] * pre_select_anchor[:, 2:],
        pre_select_anchor[:, 2:] * np.exp(box_cord[:, 2:] * VARIANCE[1])), 1)
    box_cord[:, :2] -= box_cord[:, 2:] / 2
    box_cord[:, 2:] += box_cord[:, :2]

    # calculate ldmk coordinate
    ldmk = ldmk_output[pre_select_boxes_index, :]
    ldmk[:, 0::2] = pre_select_anchor[:, 0:1] + ldmk[:, 0::2] * VARIANCE[0] * pre_select_anchor[:, 2:3]
    ldmk[:, 1::2] = pre_select_anchor[:, 1:2] + ldmk[:, 1::2] * VARIANCE[0] * pre_select_anchor[:, 3:4]

    # get prob
    box_prob = prob[pre_select_boxes_index, 1]

    return box_prob, box_cord, ldmk

def nms_oneclass(bbox: np.ndarray, score: np.ndarray, thresh: float = NMS_THRES) -> np.ndarray:

    '''
    non maximum suppression by iou
    :param bbox:
    :param score:
    :param thresh:
    :return:
    '''

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


class NetworkExecutor(object):

    def __init__(self, model_file, func):

        self.interpreter = Interpreter(model_file, num_threads=3)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.tensor_index = self.interpreter.get_input_details()[0]['index']
        self.func = func

    def get_output_tensors(self):

      output_details = self.interpreter.get_output_details()
      tensor_list = []

      for output in output_details:
            tensor = np.squeeze(self.interpreter.get_tensor(output['index']))
            tensor_list.append(tensor)

      return tensor_list

    def run(self, image):
        if image.shape[1:2] != (self.input_height, self.input_width):
            img = cv2.resize(image, (self.input_width, self.input_height))
        img = self.func(img)
        self.interpreter.set_tensor(self.tensor_index, img)
        self.interpreter.invoke()
        return self.get_output_tensors()

class FaceRecognition():
        
    def __init__(self, threshold, image_height, image_width):
        self.fd_model = NetworkExecutor("face_rec_models/ulffd_landmark.tflite", preprocess_for_fd)
        self.fe_model = NetworkExecutor("face_rec_models/inference_model_993.tflite", preprocess_for_fe)
        self.fd_input_size = [self.fd_model.input_width, self.fd_model.input_height]        
        self.resize_factors = [image_width / self.fd_model.input_width,
                               image_height / self.fd_model.input_height]
        self.threshold = float(threshold)
        self.load_db()

    def load_db(self):
        self.db = read_db("resources/database.db")

    def unregister_face(self, ID):
        return self.db.pop(ID, None)

    def draw_bounding_boxes(self, frame, detections, ids):

        color = (0, 255, 0)
        label_color = (125, 125, 125)
        pred_bbox_pixel, pred_ldmk_pixel, pred_prob = detections

        for i in range(len(pred_bbox_pixel)):
            box = [d for d in pred_bbox_pixel[i]]

            # Obtain frame size and resized bounding box positions
            frame_height, frame_width = frame.shape[:2]

            x_min, x_max = [int(position * self.resize_factors[1]) for position in box[0::2]]
            y_min, y_max = [int(position * self.resize_factors[0]) for position in box[1::2]]  

            # Ensure box stays within the frame
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

            # Draw bounding box around detected object
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Create label for detected object class
            label = 'ID: {} Name: {} {:.2f}%'.format(*ids[i])
            label_color = (255, 255, 255)

            # Make sure label always stays on-screen
            x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

            lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
            lbl_box_xy_max = (x_min + int(0.75 * x_text), y_min + y_text if y_min<25 else y_min)
            lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

            # Add label and confidence value
            cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
            cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.70, label_color, 1, cv2.LINE_AA)

            kpts = pred_ldmk_pixel[i]

            for kpt in kpts:
                cv2.circle(frame, (kpt[0], kpt[1]), 5, (255, 0, 0), 2)

    def process_faces(self, frame, detections):

        id_list = []
        pred_bbox_pixel, pred_ldmk_pixel, pred_prob = detections

        for i in range(len(pred_ldmk_pixel)):

            kpts = pred_ldmk_pixel[i]

            transformer = skimage.transform.SimilarityTransform() 
            transformer.estimate(kpts, src) 
            M = transformer.params[0: 2, : ] 
            warped_img = cv2.warpAffine(frame, M, (IMG_SHAPE[0], IMG_SHAPE[1]), borderValue = 0.0) 
            #cv2.imwrite("result.jpg", warped_img)

            features = self.fe_model.run(warped_img)[0]

            highest_score = 0
            for id in self.db.keys():
                cos_sim = np.dot(features, self.db[id]['vector'])/(np.linalg.norm(features)*np.linalg.norm(self.db[id]['vector']))
                cos_sim /= 2
                cos_sim += 0.5
                cos_sim *= 100
                if highest_score < cos_sim:
                    highest_score = cos_sim
                    recognized_id = id
                    
            if highest_score > 70.0:
                print(recognized_id, self.db[recognized_id]['name'], highest_score)
                id_list.append([recognized_id, self.db[recognized_id]['name'], highest_score])
            else:
                id_list.append(['X', '', 0.0])
        return id_list


    def register_face(self, frame, detections, registration_data):

        id, name = registration_data
        id_list = [[id, name, 100.0]]

        pred_bbox_pixel, pred_ldmk_pixel, pred_prob = detections

        if len(pred_ldmk_pixel) > 1:
            print("More than once face detected, try re-taking the picture")
            return id_list

        kpts = pred_ldmk_pixel[0]

        transformer = skimage.transform.SimilarityTransform() 
        transformer.estimate(kpts, src) 
        M = transformer.params[0: 2, : ] 
        warped_img = cv2.warpAffine(frame, M, (IMG_SHAPE[0], IMG_SHAPE[1]), borderValue = 0.0) 

        features = self.fe_model.run(warped_img)[0]

        write_db(self.db, id, name, features)
        self.load_db()

        return id_list

    def process_frame(self, frame, recognition_on, registration_data):

        start_time = time.time()
        detections = []

        bbox, ldmk, prob = self.fd_model.run(frame)

        # post processing
        pred_prob, pred_bbox, pred_ldmk = pred_boxes(bbox, prob, ldmk)

        # calculate bbox corrdinate
        pred_bbox_pixel = pred_bbox * np.tile(self.fd_input_size, 2)
        pred_ldmk_pixel = pred_ldmk * np.tile(self.fd_input_size, 5)

        # nms
        keep = nms_oneclass(pred_bbox_pixel, pred_prob)
        if len(keep) > 0:
            pred_bbox_pixel = pred_bbox_pixel[keep, :]
            pred_ldmk_pixel = pred_ldmk_pixel[keep, :]
            pred_prob = pred_prob[keep]
            pred_ldmk_pixel = ((pred_ldmk_pixel).reshape(len(keep), 5, 2)*self.resize_factors).astype(int)
            detections = pred_bbox_pixel, pred_ldmk_pixel, pred_prob

            if recognition_on:

                ids = self.process_faces(frame, detections)
                self.draw_bounding_boxes(frame, detections, ids)   

            if registration_data:
                ids = self.register_face(frame, detections, registration_data)
                self.draw_bounding_boxes(frame, detections, ids)   

        elapsed_ms = (time.time() - start_time) * 1000
        fps  = 1 / elapsed_ms*1000
        print("Estimated frames per second : {0:.2f} Inference time: {1:.2f}".format(fps, elapsed_ms))

        return frame
