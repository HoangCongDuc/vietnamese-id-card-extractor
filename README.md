# vietnamese-id-card-extractor
In this project, I try to build a library to extract information from Vietnamese ID card (the old type), both front side and back side.

Originally, I intended to let it a subproject of another project, but it is more sophisticated than I expected, I think it is necessary to make it a separate project. I worked on this project for self-learning Computer Vision. The methods used here may not be optimal, but they are the best I could do with my time and resources.

The API built in this project will take an image of the ID card, either front or back side, as the input and return a json containing all the relevant information as output. There are some constraints on the input images, generally they need not fit exactly in the image frame, but they must not deviate too much from that position. Details on this can be found in this [algorithm explanation notebook](./methods-explanation.ipynb).

The demo code can be found in [this notebook](./demo.ipynb).

In brief, the code consists of 2 mostly separate components, one for the front side and one for the back side. The overall pipeline is similar but the methods used for each step in the two are different.

The overall pipelines are as follows:
1. Align the card in the image so that it fits exactly in the image frame.
2. Crop the regions of interest in the card, those are areas in the card that contain the information.
3. Read the text from the card.

Front side methods:
- Card alignment: I use HSV values to segment the area containing the card. Then I use Hough transform to find the lines corresponding to the 4 edges. Next, I take the intersections to find 4 corners. Finally, I use perspective transform to transform the 4 corners of the card to the 4 corners of the image frame.
- Regions of interest: Because the card has fixed structure, I use fixed coordinates to locate those regions. Then I use adaptive thresholding to segment the pixels containing the text and use the result to refine the regions.
- Text reading: I used the pretrained model provided by `vietocr` to read the text.

Back side methods:
- Card alignment: I trained a segmentation model to segment the area containing the card from the image. I also use synthetic data for training. After segmentation by model, I use grabcut to refine the results, then proceed as I did with front side.
- Regions of interest: I used fixed coordiantes to locate the text regions.
- Text reading: I used pretrained model provided by `vietocr` for text reading and regular expression to post-process some results.

## Installation guide
To use this library, you need `Python 3.8` installed on your computer (earlier versions may also be used, but I used version 3.8 when I developed this library). Then, you need to install the packages in `requirements.txt`. This can be done with the following command:
```bash
pip install -r requirements.txt
```
Depending on your system, you may need to replace `pip` with `pip3`. This can be run in a [virtual environment](https://docs.python.org/3/library/venv.html).

Download the following model weight files and place it in the root directory of this repo.
- [Backside segmentation model weight](https://drive.google.com/file/d/1ErGCbLnnrw2HkUtsvTC_de1WzleP-int/view?usp=sharing).
- [`vietocr` pretrained model weight](https://drive.google.com/file/d/1LzvXrpqmRi_DuOGfoT9fRSa4_ehr0wkm/view?usp=sharing).

Now you can use this as a Python library as instructed in the [demo notebook](./demo.ipynb).

You can also run the program as a service on your local machine with the following command:
```bash
uvicorn server:app
```
By default, the service is hosted on `http://127.0.0.1:8000/`. You can open it with a web browser, upload the image, submit and see the result.

The service can also be access from terminal with `curl`.
```bash
curl -F "file=@<path/to/frontside/image>" http://127.0.0.1:8000/front/
curl -F "file=@<path/to/backside/image>" http://127.0.0.1:8000/back/
```