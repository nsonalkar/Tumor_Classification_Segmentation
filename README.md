# Tumor_Classification_Segmentation
Web App that takes a MRI scan and determines if tumor exists and where tumor is located
Data and weights:
https://drive.google.com/drive/folders/1yOxAnx9e5_VJPMZ5kLCh-vEej8pF9Z-h?usp=sharing

## Motivation
With all you can do with ML and DL techniques, being able to add value in the healthcare space is important. Being able to automatically find tumors in a brain scan would be able to reduce the time doctors take to diagnose their patients. So this is what I set out to do with this project. 

## Packages
![Data in Google Drive](https://github.com/nsonalkar/Tumor_Classification_Segmentation/blob/main/Screen%20Shot%202021-01-29%20at%204.25.12%20PM.png)
Note: Make sure tensorflow version is 2.4.1

* Download datasets and weights from shared google drive
* Make sure all files are in repository directory
* To start web app enter `streamlit run app.py` in command line

## File Description
* Tumor_classification notebook is where I created the models for this project
* app.py is a python script that builds the web app

## Web app
![alt text](https://github.com/nsonalkar/Tumor_Classification_Segmentation/blob/main/Screen%20Shot%202021-01-28%20at%2011.05.22%20PM.png)

## Acknowledgements
https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

https://www.kaggle.com/ekhtiar/resunet-a-baseline-on-tensorflow

https://setosa.io/ev/image-kernels/

https://arxiv.org/pdf/1512.03385.pdf

https://www.udemy.com/course/modern-artificial-intelligence-applications/

https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
