# Annotation-efficient-learning-for-OCT-segmentation

This repository contains the code for paper "Annotation-efficient learning for OCT segmentation". We propose an annotation-efficient learning method for OCT segmentation that could significantly reduce annotation costs and improve learning efficiency. Here we provide generative pre-trained transformer-based encoder and CNN-based segmentation decoder, both pretrained on open-access OCTdatasets. The proposed pre-trained model can be directly transfered to your ROI segmeantation based on OCT image. We hope this may help improve the intelligence and application penetration of OCT.

![Overview](images/Figure%201.png)
![Model architecture](images/Figure%202.png)

## Dependencies 
python==3.8<br>
torch==1.11.1<br>
numpy==1.19.5<br>
monai==0.7.0<br>
timm==0.3.2<br>
tensorboardX==2.1<br>
torchvision==0.12.0<br>
opencv-python==4.5.5<br>

## Usage
1. Clone the repository：
```
git clone https://github.com/SJTU-Intelligent-Optics-Lab/Annotation-efficient-learning-for-OCT-segmentation.git
```  

2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Download the pre-trained [phase1 model file](https://jbox.sjtu.edu.cn/l/d1pZvS) for weights of encoder and [phase2 model file](https://jbox.sjtu.edu.cn/l/t1vDE7) for weights of decoder, and then put them in `./runs/` folder.

4. Edit suitable path and parameters in main.py

5. Go to the corresponding folder and run:
```
cd Annotation-efficient-learning-for-OCT-segmentation
python main.py
```

## Training on your Dataset
The prepared architecture of dataset is referenced to `./dataset/` folder containing `train_fewshot_data` and `val_fewshot_data`. The name index of images is listed in `train_fewshot_data.txt` and `val_fewshot_data.txt`.

## Citation
```
@article{OSA,
  Title = {Annotation-efficient learning for OCT segmentation},
  Author = {HAORAN ZHANG, JIANLONG YANG, CE ZHENG, SHIQING ZHAO, AILI ZHANG},
  Year = {2023}
}
```
