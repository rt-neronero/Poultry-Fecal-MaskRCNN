# 🐔 Automated Poultry Disease Detection from Fecal Images Using Mask R-CNN

This repository contains the code and resources for the Poultry Disease Detection System developed using Mask R-CNN to classify and detect disease patterns in poultry fecal images. The project leverages the Mask R-CNN framework from [Matterport's Mask R-CNN repository](https://github.com/matterport/Mask_RCNN/) with custom modifications to adapt it for multi-class classification and segmentation of poultry fecal samples.

</br>

## 📚 Project Overview

Poultry farming plays a crucial role in ensuring global food security and economic stability. However, the spread of infectious diseases such as Coccidiosis, Salmonella, and Newcastle disease poses significant challenges for the industry. This project aims to automate the detection and classification of poultry diseases by analyzing fecal images using Mask R-CNN, providing an efficient and scalable solution for real-time disease monitoring and management.

</br>

## 🚀 Features

✅ Disease classification and localization from poultry fecal images. </br>
✅ Four disease classes: Coccidiosis, Salmonella, Newcastle disease, and Healthy poultry. </br>
✅ Uses Mask R-CNN for instance segmentation and object detection.</br>
✅ Configurable parameters for fine-tuning and performance optimization.</br>
✅ Generates detailed classification reports and confusion matrices for evaluation.</br>

</br>

## 📂 Dataset

The dataset used for this project was sourced from [Kaggle: Poultry Diseases Detection](https://www.kaggle.com/datasets/kausthubkannan/poultry-diseases-detection). It consists of the following original image counts:

* Coccidiosis: 2103 images
* Salmonella: 2276 images
* Newcastle disease: 376 images
* Healthy poultry: 2057 images

For this project, a **subset of 300 images per class** was selected **non-randomly** to ensure a balanced representation of each class. This subset was then used for annotation and model training.

</br>

## 📝 Annotations

* 300 images per class were selected and annotated using the [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/).
* Annotations are stored in JSON format and used as input for training the Mask R-CNN model.

</br> 

## 🧩 Dependencies

To run the project, ensure the following packages are installed:

```txt
cython==3.0.5
h5py==3.9.0
imgaug==0.4.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython-sql==0.5.0
keras==2.10
matplotlib==3.7.1
numpy==1.23.5
opencv-contrib-python==4.8.0.76
opencv-python==4.8.0.76
pillow==9.4.0
scikit-image==0.19.3
scipy==1.11.3
tensorflow-gpu==2.10
tensorboard==2.10.0
memory_profiler
imgaug
tqdm
```

</br>

## 📝 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rt-neronero/Automated-Poultry-Disease-Detection-from-Fecal-Images-Using-Mask-R-CNN
cd Automated-Poultry-Disease-Detection-from-Fecal-Images-Using-Mask-R-CNN
```

### 2. Set Up Virtual Environment

Before setting up the virtual environement, ensure that Python 3.10.11 is installed. If Python 3.10.11 is not installed, download and install it from [Python Downloads](https://www.python.org/downloads/release/python-31011/).

```bash
# Create and activate a virtual environment
virtualenv venv --python=python3.10
source venv/bin/activate   # On Linux/Mac
.\venv\Scripts\activate      # On Windows

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project Notebook

After setting up the environment and configuring the dataset, open the Jupyter Notebook to run the training process or start predicting with your own images.

</br>

## 🔍 Trained Model

You may download the trained model [here](https://github.com/rt-neronero/Poultry-Fecal-MaskRCNN/raw/refs/heads/main/mask_rcnn_poultry.h5?download=).
