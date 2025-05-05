import numpy as np
import torch
import sys
import io

sys.path.append("app/src/utils")

from ....src.func.child import get_max_boxs
from ....src.func.decrypt import decrypt_buffer

from models.experimental import attempt_load
from ....src.utils.datasets import letterbox
from ....src.utils.general import check_img_size, scale_coords, non_max_suppression
from ....src.utils.torch_utils import select_device, TracedModel


class LPDetector():  # YOLOv7
    def __init__(self, config_base):
        """
        :param weights:
        :param imgsz:
        :param device:
        """
        # Initialize
        self.IMGSZ = config_base['LP_DETECTOR']['img_size']
        self.DEVICE = select_device(config_base['DEVICE_TYPE'])
        self.HALT = self.DEVICE.type != 'cpu'
        self.AGNOSTIC_NMS = True
        self.AUGMENT = False
        self.CONF = config_base['LP_DETECTOR']['conf_thres']
        self.IOU = config_base['LP_DETECTOR']['iou_thres']
        self.CLASSES = None
        self.WEIGHTS = config_base['LP_DETECTOR']['weight']
        self.VIEW_IMG = True

        # Load model
        if self.WEIGHTS.endswith('.encrypt'):
            with open(self.WEIGHTS, 'rb') as file:
                file_buffer = file.read()
            file.close()
            encrypted_buffer = file_buffer[12:]
            nonce = file_buffer[:12]
            buffer = decrypt_buffer(nonce, encrypted_buffer)
            self.model = attempt_load(io.BytesIO(buffer), map_location=self.DEVICE)
        else:
            self.model = attempt_load([self.WEIGHTS], map_location=self.DEVICE)

        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.IMGSZ, s=self.stride)

        # To FP16
        if self.HALT:
            self.model.half()

        # Get names and colors
        if hasattr(self.model, 'module'):
            self.names = ["WHITE", "YELLOW", "BLUE", "RED", "WHITE"]
        else:
            self.names = ["WHITE", "YELLOW", "BLUE", "RED", "WHITE"]

        self.colors = [[255, 255, 255], [0, 252, 255], [252, 76, 0], [0, 50, 255], [0, 0, 0]]

        if self.DEVICE.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.DEVICE).type_as(
                next(self.model.parameters())))
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    def predict_image(self, image):
        # Padded resize
        img = letterbox(image, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.DEVICE)
        img = img.half() if self.HALT else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.AUGMENT)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   self.CONF,
                                   self.IOU,
                                   classes=self.CLASSES,
                                   agnostic=self.AGNOSTIC_NMS)
        
        # Process detections
        det = pred[0]
        list_out = []

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            for x1, y1, x2, y2, conf, cls in det.cpu().detach().numpy():
                list_out.append([int(x1),int(y1), int(x2), int(y2), float(conf), int(cls)])

        if len(list_out) > 0:
            box = get_max_boxs(list_out)
            x1, y1, x2, y2, _, _ = box
            lp_image = image[y1:y2, x1:x2]
        else:
            lp_image = None

        return lp_image
