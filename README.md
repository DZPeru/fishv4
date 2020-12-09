# fishv4 - Fish segmentation

Fish detection and segmentation based on YOLOv3 that use GrabCut to do semantic segmentation to fish market images.

# Run without install nothing

## Web version

You can try clicking [here](https://fishv4.herokuapp.com/)

# Run locally

## On your computer

Locally:

```
git clone https://github.com/DZPeru/fishv4
cd fishv4
pip3 install -r requirements.txt
```

Conda:
```
$ conda create -n fishv4 python=3.6 pip 
$ conda activate fishv4
$ pip install -r requirements.txt
```

Download the weights of the neural network to your local repository. Or do it manually, downloading from [Google Drive](https://drive.google.com/file/d/1vosRfnj3DBkZYFrzJep5_D1DV-0b6Tpv/view?usp=sharing).

```
$ gdown --output ./yolov4/fish.weights --id 1vosRfnj3DBkZYFrzJep5_D1DV-0b6Tpv
```

### 1. Command Line Approach
```
$ python main.py --image ./images/my_image.jpg --yolo yolov4
```

When finishing, you should find 8 jpg images in the project root directory.

### 2. Streamlit Approach
```
$ streamlit run app.py
```

You can upload fish market image to run the program.

The results are shown in the browser (make sure to scroll down).


## Docker version

To downloand the image and run the contaider in detach mode, run the code below.
```
$ docker container run -p 8501:8501 --rm -d pablogod/fishv4
```

To shutdown the docker type this:
```
$ docker kill <weird id of fishv4.app>
```
