from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from alignment_frontside import align_card
from alignment_backside import BacksideAligner
from reader import FrontsideReader, BacksideReader

print("Loading OCR model")
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
ocrmodel = Predictor(config)

print("Loading backside alignment model")
back_aligner = BacksideAligner()
back_aligner.load_weight('alignment_backside_weight.pth')

front_reader = FrontsideReader(ocrmodel)
back_reader = BacksideReader(ocrmodel)
print("Finish loading models")

app = FastAPI()

@app.post("/front/")
async def read_front(file: UploadFile = File(...)):
    contents = await file.read()
    bytearr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(bytearr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_aligned = align_card(img)
    info = front_reader.extract(img_aligned)
    return info

@app.post("/back/")
async def read_back(file: UploadFile = File(...)):
    contents = await file.read()
    bytearr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(bytearr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_aligned = back_aligner.warp(img)
    info = back_reader.extract(img_aligned)
    return info

@app.get("/")
async def main():
    with open('homepage.html') as f:
        content = f.read()
    return HTMLResponse(content=content)