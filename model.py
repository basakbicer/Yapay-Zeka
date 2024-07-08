# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:57:34 2023

@author: Casper
"""

from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, GRU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import TimeDistributed,Bidirectional,GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN,BatchNormalization
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score



img_height = 100
img_width = 100



def  get_images(path): 
   image_list = []
   class_list = []    
   for dirname in os.listdir(path):
    # print path to all subdirectories first.
        new_path =os.path.join(path,dirname)
        for dirname, dirnames1,filenames in os.walk(new_path):
            for filename in filenames:
                img = image_to_vector(os.path.join(new_path,filename))
                image_list.append(img)
                class_list.append(dirname)
   return np.array(image_list),class_list

def image_to_vector(img_file_path):

    img = image.load_img(img_file_path, target_size=(img_height, img_width))
    x = image.img_to_array(img)

    return x

def create_checkpoint():
    filepath = "best.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=2, #çıktı detayı
                            save_best_only=True,
                            mode='min') #en iyi model min e göre ayarlanacak
    return checkpoint

def create_cnn_model1():
    model = Sequential() # Sequential modeli oluştur
    
    model.add(Conv2D(32, (3, 3), input_shape = (img_height,img_width,3),kernel_initializer='VarianceScaling')) #2D boyutlu 32 filtreli 3*3lük matris kullanarak ölçeklemdir
    model.add(Conv2D(32, (3, 3),kernel_initializer='VarianceScaling' ))  
    model.add(BatchNormalization())
    model.add(Activation('relu')) #gizli katman 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(64, (3, 3),kernel_initializer='VarianceScaling' ))#2D boyutlu 64 filtreli 3*3lük matris kullanarak ölçeklemdir
    model.add(Conv2D(64, (3, 3),kernel_initializer='VarianceScaling' )) 
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(128, (3, 3),kernel_initializer='VarianceScaling' )) #2D boyutlu 128 filtreli 3*3lük matris kullanarak ölçeklemdir
    model.add(Conv2D(128, (3, 3),kernel_initializer='VarianceScaling' )) 
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
   
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5)) # %50 dropout uygular.aşırı öğrenmeyi engeller
    

    model.add(Flatten()) #düzleştirme katmanını ekler

    model.add(Dense(units=100, activation='sigmoid')) #100 nörona sahip tam bağlı (dense) bir katman ekler ve sigmoid aktivasyon fonksiyonunu kullanır.


    model.add(Dense(2)) #2 nörona sahip başka bir tam bağlı katman ekler ve ardından softmax aktivasyon fonksiyonunu kullanır. Bu, modelin çıkışını bir olasılık dağılımına dönüştürür.
    model.add(Activation('softmax')) 
    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model

filePath = 'C:/Users/Casper/OneDrive - bandirma.edu.tr/Masaüstü/yapayZekaa/images'  


features,classes = get_images(filePath)

features = features / 255 #normalizasyon  [0,1] piksel değerleri 255 e böl


encoder = LabelEncoder() #sınıf etiketleme
classes = encoder.fit_transform(classes)


rnd_seed = 116
np.random.seed(117) #sonuçların tekrarlanabilirliğini sağlar Bu, kodu birden çok kez çalıştırırsanız, random sneed sabit olduğu sürece aynı sonuçları almanız gerektiği anlamına gelir.
kf =StratifiedKFold(n_splits=10, random_state=rnd_seed, shuffle =True) #Kod, çapraz doğrulama için 10 katlı bir Katmanlı K-Katlama nesnesini (kf) başlatır.



scores = []
scores_auc = []
confussionMatrix = []
total = np.zeros((2,2))
filepath = "best.hdf5"


for train_index, test_index in kf.split(features,classes): 
    #result = next(kf.split(df), None)
    classes1 = to_categorical(classes, num_classes=2)
    x_train_val =features[train_index]
    x_test = features[test_index]
    y_train_val = classes1[train_index]
    y_test = classes1[test_index]
    
    x_train,x_val,y_train,y_val = sp(x_train_val, y_train_val,random_state=5, test_size=.1)
    

    model = create_cnn_model1()
  
    checkpoint = create_checkpoint()
    print('Train... Model')
    model.fit(x_train , y_train, epochs=50, batch_size=10, validation_data=(x_val,y_val),verbose=2,callbacks=[checkpoint])
    model.load_weights("best.hdf5")
    y_pred = model.predict(x_test)
    

    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    scores_auc.append(roc_auc_score(y_test, y_pred.round(),multi_class="ovr"))
    scores.append(accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    total = total + conf_mat[:2, :2]
    confussionMatrix.append(conf_mat)
print('Scores from each Iteration: ', scores)
print('Average K-Fold Score :' , np.mean(scores))

print(conf_mat)