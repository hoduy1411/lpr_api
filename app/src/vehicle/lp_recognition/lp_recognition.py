import numpy as np
import torch
import re
import io

import sys
sys.path.append("app/src/utils")

from models.experimental import attempt_load
from ....src.func.decrypt import decrypt_buffer
from ....src.utils.datasets import letterbox
from ....src.utils.general import check_img_size, scale_coords, non_max_suppression
from ....src.utils.torch_utils import select_device, TracedModel


class LPRecognition():
    def __init__(self, config_base):
        """
        :param weights:
        :param imgsz:
        :param device:
        """
        # Initialize
        self.IMGSZ = config_base['LP_RECOGNITION']['img_size']
        self.DEVICE = select_device(config_base['DEVICE_TYPE'])
        self.HALT = self.DEVICE.type != 'cpu'
        self.AGNOSTIC_NMS = True
        self.AUGMENT = False
        self.CONF = config_base['LP_RECOGNITION']['conf_thres']
        self.IOU = config_base['LP_RECOGNITION']['iou_thres']
        self.CLASSES = None
        self.WEIGHTS = config_base['LP_RECOGNITION']['weight']
        self.VIEW_IMG = True
        self.TRACE = False

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
        self.imgsz = check_img_size(config_base['LP_RECOGNITION']['img_size'], s=self.stride)

        if self.TRACE:
            self.model = TracedModel(self.model, config_base['DEVICE_TYPE']['type'], self.imgsz)

        # To FP16
        if self.HALT:
            self.model.half()

        # Get names and colors
        self.names = self.model.names

        if self.DEVICE.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.DEVICE).type_as(next(self.model.parameters())))
            
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    def predict_image(self, image, color_lp):
        (height, width, _) = image.shape

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
        # Warmup
        if self.DEVICE.type != 'cpu' and (self.old_img_b != img.shape[0] or \
                                          self.old_img_h != img.shape[2] or \
                                          self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.AUGMENT)[0]
        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.AUGMENT)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.CONF,
            self.IOU,
            classes=self.CLASSES,
            agnostic=self.AGNOSTIC_NMS
        )

        # Process detections
        det = pred[0]

        lp_number = ""

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # # Remove lower confidence level
            # list_conf_lower = []
            # for i, row in enumerate(det):
            #     if (row[4] < self.post_thres):
            #         list_conf_lower.append(i)

            # mask = torch.ones(det.size(0), dtype=bool)
            # mask[list_conf_lower] = False

            # # Use the mask to filter the rows
            # save_det = det[mask]

            det, lp_hw = normalize_potitions(det)

            one_line = lp_hw < 0.16

            if one_line :
                lp_number = self.read_lp_one_line(det, color_lp)
            else :
                lp_number = self.read_lp_two_line(det, color_lp)

            lp_number = re.sub(r'[^A-Za-z0-9]', '', lp_number.upper())
            return lp_number, float(det[-1, -2])
        else:
            return "", 0
    
    def delete_bbox_superfluous(self, det_sort_position_x1):
        arr_delete_box_idx = []
        for row_take_idx in range(len(det_sort_position_x1) - 1):
            box1 = det_sort_position_x1[row_take_idx]
            box2 = det_sort_position_x1[row_take_idx + 1]

            xi1 = max(box1[0], box2[0])
            yi1 = max(box1[1], box2[1])
            xi2 = min(box1[2], box2[2])
            yi2 = min(box1[3], box2[3])
            inter_area = (yi2 - yi1) * (xi2 - xi1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            # compute the IoU
            iou = inter_area / union_area
            if iou > 0.75:
                if box1[4] > box2[4]:
                    arr_delete_box_idx.append(row_take_idx + 1)
                else:
                    arr_delete_box_idx.append(row_take_idx)
        return arr_delete_box_idx

    def read_lp_one_line(self, det, color_lp):
        arr_lp_province_not_have = [10, 42, 44, 45, 46, 87, 91, 96]
        det_sort_position_x1 = det[det[:, 0].sort()[1]]
        arr_delete_box_idx = self.delete_bbox_superfluous(det_sort_position_x1)
        lp_number_encode = det_sort_position_x1[:, 5]
        lp_number = ""
        for character_encode in lp_number_encode:
            if int(character_encode) not in arr_delete_box_idx:
                lp_number += self.names[int(character_encode)]
        if lp_number == "":
            return lp_number
        # Check color with format lp
        # if color_lp == 3:
        #     if lp_number.isnumeric() is True or len(lp_number) < 6 or len(lp_number) > 9 \
        #             or lp_number[0].isnumeric() is True or lp_number[1].isnumeric() is True \
        #             or lp_number[2].isnumeric() is False:
        #         return ""

        if color_lp == 0 or color_lp == 1 or color_lp == 2:
            if lp_number.isnumeric() is True or len(lp_number) <= 5 or len(lp_number) > 11:
                return ""
            # if check_format_license_plate(lp_number) == True:
            #     index_word = int(re.search('\D', lp_number).span()[0])
            #     lp_number = lp_number[0:index_word + 1] + "-" + lp_number[index_word + 1:]
            #     if len(lp_number[index_word + 1:].replace("-", "")) > 5:
            #         return ""
            return lp_number
        else:
            # if re.search('\D', lp_number) == None:
            #     return ""
            return lp_number


    def read_lp_two_line(self, det, color_lp):
        arr_lp_province_not_have = [10, 42, 44, 45, 46, 87, 91, 96]
        list_character_top = []
        list_character_bottom = []
        lp_number = ""
        lp_number_base = ""
        # Decode class
        lp_number_above = ""
        lp_number_below = ""

        det = det.cpu().numpy()
        y1_min = det[:, 1].min()
        y1_max = det[:, 1].max()
        y1_mean = (y1_max + y1_min) / 2

        for line in det:
            if line[1] < y1_mean:
                list_character_top.append(line)
            else:
                list_character_bottom.append(line)

        if len(list_character_top) > 1 and len(list_character_bottom) > 1:
            # Convert list to array
            array_character_bottom = np.asarray(list_character_bottom)
            array_character_top = np.asarray(list_character_top)
            # Sort ascending x1
            array_top_sort_position_x1 = array_character_top[array_character_top[:, 0].argsort()]
            array_bottom_sort_position_x1 = array_character_bottom[array_character_bottom[:, 0].argsort()]

            arr_delete_box_idx_top = self.delete_bbox_superfluous(array_top_sort_position_x1)
            arr_delete_box_idx_botom = self.delete_bbox_superfluous(array_bottom_sort_position_x1)

            array_character_top = array_top_sort_position_x1[:, 5]
            array_character_bottom = array_bottom_sort_position_x1[:, 5]

            for character_encode in array_character_top:
                if int(character_encode) not in arr_delete_box_idx_top:
                    lp_number_above += self.names[int(character_encode)]

            for character_encode in array_character_bottom:
                if int(character_encode) not in arr_delete_box_idx_botom:
                    lp_number_below += self.names[int(character_encode)]
            # if len(lp_number_below) < 4:
            #     return ""
            lp_number = lp_number_above + "-" + lp_number_below
            lp_number_base = lp_number_above + lp_number_below
        # Check color with format lp
        # if color_lp == 3:
        #     if lp_number_base.isnumeric() is True or len(lp_number_base) < 6 or len(lp_number_base) > 9 \
        #             or lp_number_base[0].isnumeric() is True or lp_number_base[1].isnumeric() is True \
        #             or lp_number_base[2].isnumeric() is False:
        #
        #         return ""
        if color_lp == 0 or color_lp == 1 or color_lp == 2:
            if lp_number_base.isnumeric() is True or len(lp_number_above) > 6 or len(lp_number_below) > 7 or len(lp_number_below) <= 3:
                return ""
            if check_format_license_plate(lp_number.replace("-", "")) == True:
                # if lp_number[0] == "0":
                #     lp_number = list(lp_number)
                #     lp_number[0] = "Q"
                #     lp_number = "".join(lp_number)
                return lp_number
        else:
            # if re.search('\D', lp_number.replace("-", "")) == None:
            #     return ""
            return lp_number
        return lp_number


def check_format_license_plate(lp_number):
    bicycle = re.match(r'(\d{2})(\D{2})(\d{4})', lp_number)
    moto_1 = re.match(r'(\d{2})(\D{2})(\d{5})', lp_number)
    moto_2 = re.match(r'(\d{1})(\D{1})(\d{4})', lp_number)
    moto_3 = re.match(r'(\d{1})(\D{1})(\d{5})', lp_number)
    moto_4 = re.match(r'(\d{2})(\D{1})(\d{5})', lp_number)
    moto_5 = re.match(r'(\d{2})(\D{1})(\d{6})', lp_number)
    # moto_6 = re.match(r'(\d{2})(\w{1})(\d{6})', lp_number)
    oto = re.match(r'(\d{2})(\D{1})(\d{4})', lp_number)

    if bicycle or moto_1 or moto_2 or moto_3 or moto_4 or \
            moto_5 or oto:
    # if bicycle or moto_1 or moto_3 or moto_4 or \
    #         moto_5 or oto :
        return True
    return False


def normalize_potitions(det):

    if det is None or det.numel() == 0:
        return det, 0.0

    # Extract x1, y1 for least squares fitting
    x1_coords = np.array([char[0] for char in det])
    y1_coords = np.array([char[1] for char in det])

    # Calculate the least squares line
    A = np.vstack([x1_coords, np.ones(len(x1_coords))]).T
    m, c = np.linalg.lstsq(A, y1_coords, rcond=None)[0]

    # Calculate the angle of rotation
    theta = -np.arctan(m)

    # Create the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])

    # Rotate each character
    rotated_chars = []
    for char in det:
        x1, y1, x2, y2, conf, name = char
        original_coords = np.array([[x1, y1], [x2, y2]]).T
        rotated_coords = R @ original_coords
        char[1] = round(rotated_coords[1, 0])
        char[2] = round(rotated_coords[0, 1])
        char[3] = round(rotated_coords[1, 1])
        char[0] = round(rotated_coords[0, 0])

    x1_max = det[:, 0].max()
    x1_min = det[:, 0].min()
    y1_min = det[:, 1].min()
    y1_max = det[:, 1].max()

    lp_hw = (y1_max-y1_min)/(x1_max-x1_min)

    # print(f'len = {x1_max-x1_min} {y1_min} {y1_max} {lp_hw}')
    return det, lp_hw