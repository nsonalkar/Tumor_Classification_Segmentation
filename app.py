import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os



def tversky(y_true, y_pred, smooth = 1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def main():
    st.title("Brain Tumor Image Segmentation")


    activities = ["Show Image"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice == "Show Image":
        st.subheader("Brain Image")
        img = st.file_uploader("Upload Brain Scan Image")
        #print(img.read())
        #plt.imshow(cv2.imread(img))
        if img:
            img.seek(0)
            st.image(np.array(Image.open(img)))
            with open("./resnet-50-MRI.json","r") as json_file:#classifier-resnet-model.json
                json_savedModel = json_file.read()
            model = tf.keras.models.model_from_json(json_savedModel)
            model.load_weights('./weights.hdf5')
            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            with open("./segmentation-resunet-model.json","r") as json_file:
                json_savedModel= json_file.read()
            model_seg = tf.keras.models.model_from_json(json_savedModel)
            model_seg.load_weights('./weights_seg.hdf5')
            adam = tf.keras.optimizers.Adam(lr = 0.05, epsilon = 0.1)
            model_seg.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])
            try:
                img2 = np.array(Image.open(img))
                img = np.array(Image.open(img))
                img = img * 1./255.

                #Reshaping the image
                img = cv2.resize(img,(256,256))
                #Converting the image into array
                img = np.array(img, dtype = np.float64)

                #reshaping the image from 256,256,3 to 1,256,256,3
                img = np.reshape(img, (1,256,256,3))

                #making prediction on the image
                is_defect = model.predict(img)
                if np.argmax(is_defect) == 0:
                    st.write("No Tumor")
                else:
                    X = np.empty((1, 256, 256, 3))
                    img = cv2.resize(img2,(256,256))
                    print(img)
                    img = np.array(img, dtype = np.float64)
                    #standardising the image
                    img -= img.mean()
                    img /= img.std()

                    #converting the shape of image from 256,256,3 to 1,256,256,3
                    X[0,] = img
                    predict = model_seg.predict(X)

                    img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    # st.image(np.asarray(img))
                    # st.image(np.asarray(predict)[0].squeeze().round())
                    img_ = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    img_[np.asarray(predict)[0].squeeze().round() == 1] = (0,255,0)
                    st.image(img_)
                    st.write("Tumor")
            except:
                st.write("Brain Scan Isn't Clear")


if __name__ == "__main__":
    main()
