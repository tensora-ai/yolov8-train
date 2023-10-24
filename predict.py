import ast
import math
import logging

import cv2
import numpy as np
import onnxruntime as ort


def load_img(source: str | bytes | np.ndarray) -> np.ndarray:
    if isinstance(source, bytes):
        arr = np.frombuffer(source, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if isinstance(source, str):
        return cv2.imread(source, cv2.IMREAD_COLOR)
    assert isinstance(source, np.ndarray) and len(source.shape) == 3
    return source


def is_blurry(img, size=1280, threshold=2.0):
    img = load_img(img)
    h, w, c = img.shape
    s = size / (h if h > w else w)
    img2 = cv2.resize(img, (int(s * w), int(s * h)))
    val = math.log10(cv2.Laplacian(img2, ddepth=6).var())
    return val < threshold


class Yolov8:
    def __init__(self, onnx_model, blur_check=False, conf_thres=0.5, iou_thres=0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.blur_check = blur_check

        self.initialize(onnx_model)

    def initialize(self, onnx_model):
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_model, providers=providers)
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.trg_w = inputs[0].shape[3]
        self.trg_h = inputs[0].shape[2]
        # Important for current resize logic
        assert self.trg_w == self.trg_h

        try:
            meta = self.session.get_modelmeta()
            classes = meta.custom_metadata_map["names"]
            classes = ast.literal_eval(classes)  # str to dict
            self.classes = [classes[key] for key in sorted(classes.keys())]
        except:
            logging.warning("Class names weren't found in the model metadata")
            self.classes = [str(i) for i in range(80)]  # 80 == n classes COCO
        self.n_classes = len(self.classes)

    def __call__(self, source: str | bytes | np.ndarray) -> tuple:
        return self.detect(source)

    def detect(self, source: str | bytes | np.ndarray) -> tuple:
        image = load_img(source)
        tensor = self.preprocess(image)
        output = self.inference(tensor)
        boxes, scores, ids = self.postprocess(output)

        logging.info(f"Detected {len(boxes)} objects.")

        return boxes, scores, ids, self.classes

    def resize(self, image):
        self.src_h, self.src_w = image.shape[:2]
        scale = self.trg_w / max(self.src_w, self.src_h)
        image = cv2.resize(image, (int(scale * self.src_w), int(scale * self.src_h)))
        h, w = image.shape[:2]
        self.pad_x = (self.trg_w - w) // 2
        self.pad_y = (self.trg_h - h) // 2
        arr = np.zeros((self.trg_h, self.trg_w, 3))
        arr[self.pad_y : self.pad_y + h, self.pad_x : self.pad_x + w] = image
        return arr

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize(image) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image[None, ...].astype(np.float32)

    def inference(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]

    def postprocess(self, output: np.ndarray) -> tuple:
        preds = np.squeeze(output).T
        boxes, scores, ids = [], [], []
        trg2src = max(self.src_h, self.src_w) / self.trg_w

        for pred in preds:
            classes_scores = pred[4:]
            conf = np.amax(classes_scores)

            if conf >= self.conf_thres:
                class_id = np.argmax(classes_scores)

                # Absolute target coordinates in XYWH format
                x, y, w, h = pred[0], pred[1], pred[2], pred[3]

                # Adjust for padding
                x, y = x - self.pad_x, y - self.pad_y

                # Absolute source coordinates in XYWH format
                x, y, w, h = [e * trg2src for e in (x, y, w, h)]

                # Absolute source coordinates in LTWH format
                l, t = x - w / 2, y - h / 2

                # Add the class ID, score, and box coordinates to the respective lists
                ids.append(class_id)
                scores.append(conf)
                boxes.append([l, t, w, h])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)

        boxes = [[round(k) for k in boxes[i]] for i in indices]
        scores = [round(float(scores[i]), 3) for i in indices]
        ids = [int(ids[i]) for i in indices]

        return boxes, scores, ids
