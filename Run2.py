#Importing Necessary packages

import pandas as pd
import cv2
import numpy as np
import os
import random
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from collections import Counter
from random import sample
from sklearn.metrics import classification_report




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""#1.Reading the Image"""

#Function to read image and resize image
def read_image(label,num):
    path = 'training/training'
    img = cv2.imread(path+"/"+label+"/"+num+".jpg",cv2.IMREAD_GRAYSCALE)
    resize_img = cv2.resize(img,(256,256))
    return resize_img




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""#2.Extracting Features"""
#Function to extract feature from an image
def feature_extraction(image):
    feature=[]

    temp = np.zeros((8,8), dtype=float)
    for j in range(0,image.shape[1],4):
        for i in range(0,image.shape[0],4):
            temp = image[i:i+8,j:j+8]

            if (temp.shape[0]==8 and temp.shape[1]==8 ):
                temp = temp.flatten()
                temp = normalisation(temp)
                feature.append(temp)
    return feature

#Function to normalise an array- here an image
def normalisation(array):
    mean = array.mean()
    std_dev = array.std()
    b = np.array(std_dev)

    array = array - mean
    array = array/255
    return array




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""#3.Creating Dataset"""
# Function to create dataset

def dataset(path):
    dir_list = os.listdir(path)
    features=[] # list to store feature value
    labels=[] # list to store label value

    for i in dir_list:
        for j in range(0,100):
            print(i," ",j)
            image = read_image(i,str(j))  #Calling read_image function
            feature = feature_extraction(image) #Callingfeature extratcion function

            features.append(feature)
            labels.append(i)

    return features,labels


path = 'training/training'
feature_set,label_set = dataset(path)




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""#4. Train-Validation Split"""

l = [x for x in range(0, len(feature_set))]  #temporary list to store the number of images in feature set
train_ind = random.sample(range(0, len(feature_set)), int(0.70*len(feature_set))) # Identifying indexes for training set
val_ind = [x for x in l if x not in train_ind] # Identifying indexes for validation set

X_train = [feature_set[i] for i in train_ind] #Creating Training set
X_val = [feature_set[i] for i in val_ind] #Creating validation set
y_train = [label_set[i] for i in train_ind]
y_val = [label_set[i] for i in val_ind]




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""#5. Learning Vocabulary"""

# Function to learn vocabulart i.e running kmeas algo
def learning_words(feature_set):
    kmeans = KMeans(n_clusters=500, random_state=0, n_init="auto").fit(feature_set) #Creating Kmeans model to learn vocabulary and returning the model
    return kmeans


# Function to map the learnt Vocabulary to the training dataset i.e Creating histogram 
def mapping(kmeans,df):
    num = [x for x in range(0,500)] #500 is the number of clusters
    mapped_df = pd.DataFrame(columns=num)

    k=0
    for i in df:
        print(k)
        c = kmeans.predict(i)
        t = Counter(c)
        new_dict = dict(t)
        mapped_df = mapped_df.append(new_dict, ignore_index=True)
        k=k+1

    mapped_df = mapped_df.fillna(0)
    return mapped_df


# Making dataset for learning vocab
feature_vocab=[]
for i in X_train:
    for j in i:
        feature_vocab.append(j)

# Shuffelling the Dataset
random.shuffle(feature_vocab)

# Extracting Kmeans model which learns the vocabulary
kmeans = learning_words(feature_vocab[:int(len(feature_vocab)*0.25)])

mapped_df_train = mapping(kmeans,X_train) #Mapping dataset with vocabulary
mapped_df_train['label']=y_train




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""#6. Creating Classifiers- 1 linear classifiers for each repective class"""

#Function to return a logisitic regression model 
def classifier(mapped_df,class_label):

    #Getting features for 100 images for a particular class
    class_df=mapped_df[mapped_df['label']==class_label]
    class_df['label']=1

    #Getting features for 100 images from miz of other classes
    non_class_df = mapped_df[mapped_df['label']!=class_label]
    non_class_df['label']=0

    ttrain =  class_df.append(non_class_df.sample(n=100)) #Creating temporary df for training set

    #X y splt
    y= ttrain['label']
    X= ttrain.drop(columns='label')

    clf = LogisticRegression(random_state=0).fit(X, y) #Running a logistice regression model
    return clf

classes= list(set(mapped_df_train['label'])) #getting a list of different classes= 15 scenes
classifiers = [] #List to store model for each class

#Loop to store classifiers in a list
for i in classes:
    classify = classifier(mapped_df_train,i)
    classifiers.append(classify)




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""#7. Making Predictions"""

#Function to return predicted values from all the classifier (taking an average from all predictions)
def predict(mapped_df,classifiers):
    predictions = pd.DataFrame() #Dataframe to store predictions from all classesifers
    k=0
    for i in classifiers:
        predictions[classes[k]]= i.predict_proba(mapped_df.drop(columns='label'))[:,1]
        k=k+1

    maxValueIndex = predictions.idxmax(axis=1) #Identying class with highest probability 
    predictions['final_label'] = maxValueIndex #Assigning final prediction

    y_pred = predictions['final_label']

    return y_pred


"""# Making Predicitions Using Training set"""
y_train_pred = predict(mapped_df_train,classifiers)
print(classification_report(y_train,y_train_pred))

"""# Making Predicitions Using Validation set"""
mapped_df_val = mapping(kmeans,X_val) #Creating histogram using vocabulary for validation set
mapped_df_val['label']=y_val
y_val_pred = predict(mapped_df_val,classifiers)
print(classification_report(y_val,y_val_pred))




"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 8. Making predicitions on test set"""

#Function to read image from a path
def read_image_test(path,num):
    img = cv2.imread(path+"/"+num,cv2.IMREAD_GRAYSCALE)
    resize_img = cv2.resize(img,(256,256))
    return resize_img

#Function to create dataset for testing set
def dataset_test(path):

    dir_list = os.listdir(path)
    features=[]
    labels=[]

    for i in dir_list:
        print(i)
        image = read_image_test(path,i) #Calling read_image function
        feature = feature_extraction(image) #Calling feature_extraction function

        features.append(feature)
        labels.append(i)

    return features,labels


test_path = 'testing/testing'
feature_set,label_set = dataset_test(test_path) #Creating testing dataset

mapped_test_set = mapping(kmeans,feature_set) #Creating histogram using vocabulary for validation set
mapped_test_set['label']=0
y_test_pred = predict(mapped_test_set,classifiers) #Making Prediciton on test set



#Storing predictions in a datframe for testing dataset
lists =[m+" "+n for m,n in zip(label_set,y_test_pred)]
re=pd.DataFrame(lists,columns=['Name'])
re['Num']= [int(x.split('.')[0]) for x in re['Name'] ]
re = re.sort_values(by=['Num'])
re = re.reset_index()
sub = list(re['Name'])

#Saving the predictions in a txtfile
with open(r'run2.txt', 'w') as fp:
    for item in sub:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')