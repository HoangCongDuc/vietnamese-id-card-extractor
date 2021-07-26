from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
from alignment_frontside import align_card
from alignment_backside import BacksideAligner
from reader import FrontsideReader, BacksideReader

print("Loading frontside reader model")
front_reader = FrontsideReader()

print("Loading backside alignment model")
back_aligner = BacksideAligner()
back_aligner.load_weight('alignment_backside_weight.pth')

print("Loading backside reader model")
back_reader = BacksideReader()
print("Finish loading models")

app = FastAPI()

@app.post("/front/")
async def read_front(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img = np.asarray(img)
    img_aligned = align_card(img)
    info = front_reader.extract(img_aligned)
    return info

@app.post("/back/")
async def read_back(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img = np.asarray(img)
    img_aligned = back_aligner.warp(img)
    info = back_reader.extract(img_aligned)
    return info

@app.get("/")
async def main():
    with open('homepage.html') as f:
        content = f.read()
    return HTMLResponse(content=content)