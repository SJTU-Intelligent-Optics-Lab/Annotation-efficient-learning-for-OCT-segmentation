# Annotation-efficient-learning-for-OCT-segmentation

This repository contains the code for the paper ["Annotation-efficient learning for OCT segmentation"](https://opg.optica.org/boe/fulltext.cfm?uri=boe-14-7-3294&id=531648). We propose an annotation-efficient learning method for OCT segmentation that could significantly reduce annotation costs and improve learning efficiency. Here we provide generative pre-trained transformer-based encoder and CNN-based segmentation decoder, both pretrained on open-access OCTdatasets. The proposed pre-trained model can be directly transfered to your ROI segmeantation based on OCT image. We hope this may help improve the intelligence and application penetration of OCT.

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
3. Download the pre-trained [phase1 model file](https://drive.google.com/file/d/1JHdL1HRZM86n4761uoO3_6N4p4zqBDmx/view?usp=sharing) for weights of encoder and [phase2 model file](https://drive.google.com/file/d/1gOihHsH4-GAtS6R6wzxkOQsaMAF6fj_h/view?usp=sharing) for weights of decoder, and then put them in `./runs/` folder.

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
@article{
  title={Annotation-efficient learning for OCT segmentation},
  author={Zhang, Haoran and Yang, Jianlong and Zheng, Ce and Zhao, Shiqing and Zhang, Aili},
  journal={Biomedical Optics Express},
  volume={14},
  number={7},
  pages={3294--3307},
  year={2023},
  publisher={Optica Publishing Group}
}
```
