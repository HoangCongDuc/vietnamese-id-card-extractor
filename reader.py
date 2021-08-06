import re
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


class FrontsideReader:
    def __init__(self, ocrmodel):
        self.text_reader = ocrmodel

    def _get_text(self, img):
        if img is None:
            return ''
        img_pil = Image.fromarray(img)
        return self.text_reader.predict(img_pil)

    def _concat(self, text1, text2, sep):
        if len(text1) == 0:
            return text2
        if len(text2) == 0:
            return text1
        return sep.join([text1, text2])

    def _vertical_offset(self, img_bin):
        intensity = np.mean(img_bin, axis=1)
        intensity = gaussian_filter1d(intensity, sigma=5, mode='reflect')
        
        mid = intensity.size // 2
        top = np.argmin(intensity[:mid])
        bot = np.argmin(intensity[:mid:-1])
        return top, bot

    def _horizontal_intensity(self, img_bin):
        intensity = np.mean(img_bin, axis=0)
        intensity_gauss = gaussian_filter1d(intensity, sigma=25, mode='constant')
        return intensity_gauss

    def detect(self, img):
        box = [
            [170, 240, 340],
            [215, 285, 290],
            [255, 325, 380],
            [295, 365, 420],
            [345, 415, 250],
            [385, 455, 515],
            [425, 495, 250]
        ]
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_bin = cv2.adaptiveThreshold(src=img_gray,
                                        maxValue=255,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                        thresholdType=cv2.THRESH_BINARY_INV,
                                        blockSize=11,
                                        C=20)

        for i in range(7):
            img_text_bin = img_bin[box[i][0]:box[i][1], box[i][2]:]
            top, bot = self._vertical_offset(img_text_bin)
            box[i][0] = box[i][0] + top
            box[i][1] = box[i][1] - bot + 1
            
        intensity = [self._horizontal_intensity(img_bin[box[i][0]:box[i][1], box[i][2]:]) for i in range(7)]
        max_intensity = [np.max(intensity[i]) for i in range(7)]
        M = max(max_intensity)
        is_empty = [max_intensity[i] < M/2 for i in range(7)]

        img_text = [img[120:175, 390:700]]
        for i in range(7):
            if is_empty[i]:
                img_text.append(None)
            else:
                text_idx = np.argwhere(intensity[i] >= max_intensity[i] / 2).squeeze()
                left = max(0, np.min(text_idx - 20))
                right = np.max(text_idx) + 21
                img_text.append(img[box[i][0]:box[i][1], box[i][2]+left:box[i][2]+right])

        return img_text

    def extract(self, img):
        img_text = self.detect(img)
        number = self._get_text(img_text[0])
        
        name1 = self._get_text(img_text[1])
        name2 = self._get_text(img_text[2])
        name = self._concat(name1, name2, ' ')

        dob = self._get_text(img_text[3])

        origin1 = self._get_text(img_text[4])
        origin2 = self._get_text(img_text[5])
        origin = self._concat(origin1, origin2, ', ')

        address1 = self._get_text(img_text[6])
        address2 = self._get_text(img_text[7])
        address = self._concat(address1, address2, ', ')
        
        output = {
            "Số CMND": number,
            "Họ tên": name,
            "Sinh ngày": dob,
            "Nguyên quán": origin,
            "Nơi ĐKHK thường trú": address
        }
        return output


class BacksideReader:
    def __init__(self, ocrmodel):
        self.text_reader = ocrmodel

    def _get_text(self, img):
        img_pil = Image.fromarray(img)
        return self.text_reader.predict(img_pil)

    def parse_text(self, text):
        pattern = "[^\d]*(\d{1,2})[^\d]*(\d{1,2}).*?(\d+)"
        match = re.fullmatch(pattern, text)
        return match.group(1), match.group(2), match.group(3)

    def extract(self, img):
        ethnic = self._get_text(img[10:75, 130:400])
        religion = self._get_text(img[10:75, 550:750])
        trace1 = self._get_text(img[125:180, 320:])
        trace2 = self._get_text(img[175:230, 320:])
        trace = ' '.join([trace1, trace2])
        date_issue = self._get_text(img[240:290, 325:])
        date_issue = self.parse_text(date_issue)
        place_issue = self._get_text(img[285:335, 500:])

        output = {
            "Dân tộc": ethnic,
            "Tôn giáo": religion,
            "Dấu vết riêng và dị hình": trace,
            "Ngày cấp": {
                "Ngày": int(date_issue[0]),
                "Tháng": int(date_issue[1]),
                "Năm": int(date_issue[2])
            },
            "Nơi cấp": place_issue
        }
        
        return output