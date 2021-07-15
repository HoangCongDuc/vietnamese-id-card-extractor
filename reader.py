import json, re
import cv2
import frontside_model

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


class FrontsideReader:
    def __init__(self):
        with open('frontside_reader.config', 'r') as f:
            config = json.load(f)
        self.text_reader = frontside_model.Predictor(config['model'], 'frontside_reader.pth')
    
    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_text(self, img):
        new_h = 32
        h = img.shape[0]
        img = cv2.resize(img, (0, 0), fx=new_h/h, fy=new_h/h)
        img = img / 255
        return self.text_reader.predict(img)

    def _concat(self, text1, text2, sep):
        if len(text1) == 0:
            return text2
        if len(text2) == 0:
            return text1
        return sep.join([text1, text2])

    def extract(self, img):
        number = self._get_text(img[120:175, 390:700])
        
        name = self._get_text(img[170:220,340:])

        dob = self._get_text(img[260:310,380:])

        origin1 = self._get_text(img[305:355, 420:])
        origin2 = self._get_text(img[350:400, 250:])
        origin = self._concat(origin1, origin2, ', ')

        address1 = self._get_text(img[395:445, 515:])
        address2 = self._get_text(img[435:485, 250:])
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
    def __init__(self):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = './transformerocr.pth'
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        config['predictor']['beamsearch']=False

        self.text_reader = Predictor(config)

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
            # "Ngày cấp": date_issue,
            "Nơi cấp": place_issue
        }
        
        return output