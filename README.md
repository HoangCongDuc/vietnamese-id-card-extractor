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
- Regions of interest: Because the card has fixed structure, I use fixed coordinates to locate those regions.
- Text reading: I retrain a model from `vietocr` to read texts with background from the ID card. I have essentially no real data, so I synthesize data to train it.

Back side methods:
- Card alignment: I trained a segmentation model to segment the area containing the card from the image. I also use synthetic data for training. After segmentation by model, I use grabcut to refine the results, then proceed as I did with front side.
- Regions of interest: Similar to front side.
- Text reading: I used pretrained model provided by `vietocr` for text reading and regular expression to post-process some results.

You will need these files to run the code.
- [Front side text reader model weight](https://drive.google.com/file/d/14cGUnx7xEs0PHtwEMkDi6XeBvMtCUpJs/view?usp=sharing).
- [Backside segmentation model weight](https://drive.google.com/file/d/1ErGCbLnnrw2HkUtsvTC_de1WzleP-int/view?usp=sharing).
- [`vietocr` pretrained model weight, used for back side text reading](https://drive.google.com/file/d/1LzvXrpqmRi_DuOGfoT9fRSa4_ehr0wkm/view?usp=sharing).

## Reference
1. [Technical report of a similar project from a group of students in VNU University of Engineering and Technology, Hanoi](https://eprints.uet.vnu.edu.vn/eprints/id/eprint/3281/2/technical_report_v1_Cong_Tuan.pdf).
2. https://alessandroosias.com/automatic-id-card-information-extraction/
3. [Technical review of FPT card reader](https://fpt.ai/technical-view-fvi-end-end-vietnamese-id-card-ocr)
4. https://github.com/eragonruan/text-detection-ctpn/tree/banjin-dev
5. [Real-time information retrieval from Identity cards](https://arxiv.org/pdf/2003.12103.pdf)
6. https://pbcquoc.github.io/vietocr/