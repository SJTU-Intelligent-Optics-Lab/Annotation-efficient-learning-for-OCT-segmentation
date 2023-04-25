# Annotation-efficient-learning-for-OCT-segmentation

This repository contains the code for paper "Annotation-efficient learning for OCT segmentation". We propose an annotation-efficient learning method for OCT segmentation that could significantly reduce annotation costs and improve learning efficiency. Here we provide generative pre-trained transformer-based encoder and CNN-based segmentation decoder, both pretrained on open-access OCTdatasets. The proposed pre-trained model can be directly transfered to your ROI segmeantation based on OCT image. We hope this may help improve the intelligence and application penetration of OCT.

![示例图片](images/Figure 1.png)

## Installing Dependencies 
python==3.8<br>
torch==1.11.1<br>
numpy==1.19.5<br>
monai==0.7.0<br>
timm==0.3.2<br>
tensorboardX==2.1<br>
torchvision==0.12.0<br>
opencv-python==4.5.5<br>
