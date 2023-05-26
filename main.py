import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Dropout,MaxPooling2D,GlobalAvgPool2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
np.random.seed = 32

def import_data() :
    image_dir = "pizza_not_pizza/"

    not_pizza = [(os.path.join(image_dir,"not_pizza",image),0) for image in os.listdir(os.path.join(image_dir,"not_pizza")) if image.split(".")[1] == "jpg"]
    pizza = [(os.path.join(image_dir,"pizza",image),1) for image in os.listdir(os.path.join(image_dir,"pizza")) if image.split(".")[1] == "jpg"]

    df = pd.DataFrame(not_pizza+pizza,columns=['filename','category'])
    df.sample()

    train_df,dummy_df = train_test_split(df,train_size=0.7,random_state=42,shuffle=True)
    val_df,test_df = train_test_split(dummy_df,train_size=0.6,random_state=42,shuffle=True)

    train_df.category = train_df.category.astype(str)
    val_df.category = val_df.category.astype(str)
    test_df.category = test_df.category.astype(str)

    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = datagen.flow_from_dataframe(train_df,x_col='filename',y_col='category',target_size=(256,256),batch_size=32,class_mode="binary",shuffle=True)
    val_generator = datagen.flow_from_dataframe(val_df,x_col='filename',y_col='category',target_size=(256,256),batch_size=32,class_mode="binary",shuffle=True)
    test_generator = datagen.flow_from_dataframe(test_df,x_col='filename',y_col='category',target_size=(256,256),batch_size=32,class_mode="binary",shuffle=True)
    return train_generator,val_generator,test_generator

def create_model() :
    base_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(256,256,3))
    for layer in base_model.layers:
        layer.trainable=False

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAvgPool2D())
    model.add(Dense(512,activation='relu',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))

    model.build(input_shape=(None, 256, 256, 3))
    model.summary()

    model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
    return model

def train_model(model,train_generator,val_generator) :
    modelcheck = ModelCheckpoint(filepath='model.h5',monitor='val_loss',save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
    tensorboard = TensorBoard(log_dir='logs')

    history = model.fit(train_generator,validation_data=val_generator,epochs=20,callbacks=[modelcheck,earlystop,tensorboard])

    return history


def test_model(model,test_generator) :
    model.evaluate(test_generator)
    
def graphics(history) :
    fig,ax = plt.subplots(1,2,figsize=(15,9))
    total_epochs = [i for i in range(len(history.history['loss']))]
    fig.suptitle("CNN Performance")

    ax[0].plot(total_epochs,history.history['loss'],label='train')
    ax[0].plot(total_epochs,history.history['val_loss'],label='val')
    ax[0].set_title("Training Loss vs Validation Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss %")
    ax[0].legend(loc='best')

    ax[1].plot(total_epochs,history.history['accuracy'],label='train')
    ax[1].plot(total_epochs,history.history['val_accuracy'],label='val')
    ax[1].set_title("Training Accuracy vs Validation Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy %")
    ax[1].legend(loc='best')
    plt.show()



    
