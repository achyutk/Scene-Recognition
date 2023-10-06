import pandas as pd
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from collections import Counter
from random import sample



"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
'''Functions for run1'''

def feature_extraction_r1(paths):
    features = []
    for row in paths:
        # read image from img_path then convert to gray scale
        image = Image.open(row)
        img_width, img_height = image.size
        # crop image at center
        image = image.crop(((img_width - 200) // 2,(img_height - 200) // 2,
                            (img_width + 200) // 2,(img_height + 200) // 2))
        # resize image to image with height and width 16
        image = image.resize((16,16))
        # convert to array
        image = np.array(image)
        # standardize images ensuring all pixels values have mean of zero and standard deviation of 1
        image = (image - image.mean()) / 255.0
        # reshape the images to 16x16
        image = image.reshape(16*16)
        # append the final processed image to data list
        features.append(image)
    return features








"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
'''Functions for run2'''



## 1.Reading the Image
def read_image(label,num):
    path = 'training/training'
    img = cv2.imread(path+"/"+label+"/"+num+".jpg",cv2.IMREAD_GRAYSCALE)
    resize_img = cv2.resize(img,(256,256))
    return resize_img



"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""

## 2.Extracting Features

# Extracting Features
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

def normalisation(array):
    mean = array.mean()
    std_dev = array.std()
    b = np.array(std_dev)


    array = array - mean  
    array = array/255
    return array 

"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""

## 3.Creating Dataset

# Function to create dataset
def dataset(path):

    dir_list = os.listdir(path)
    
    features=[]
    labels=[]
    
    for i in dir_list:
        for j in range(0,100):
            print(i," ",j)
            image = read_image(i,str(j))
            feature = feature_extraction(image)
            
            features.append(feature)
            labels.append(i)
            


    return features,labels


"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""

## 4. Learning Vocabulary
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

"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""


## 6. Creating Classifiers- 1 linear classifiers for each repective class
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

"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""

## 7. Making Predictions
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

"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""

## 8. Making predicitions on test set
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



"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
'''Functions for run3'''


# SIFT Visual words
def sift(image, step_size=5):
    # construct a SIFT object 
    sift = cv2.SIFT_create()
    # find keypoints and descriptors
    kp, des = sift.detectAndCompute(image, None)
    
    return des

def dense_sift(image, step_size=5):
    # construct a SIFT object 
    sift = cv2.SIFT_create()
    # create dense keypoints and compute descriptors
    kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size) 
                                        for y in range(0, image.shape[1], step_size)]
    kp, des = sift.compute(image, kp)
    return des

def dense_orb(image, step_size=5):
    # read image from img_path then convert to gray scale
    # construct a ORB object 
    orb = cv2.ORB_create()
    # create dense keypoints and compute descriptors
    kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size) 
                                        for y in range(0, image.shape[1], step_size)]
    kp, des = orb.compute(image, kp)
    return np.float32(des)

def dense_brief(image, step_size=5):
    # construct a BRIEF object 
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # create dense keypoints and compute descriptors
    kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size) 
                                        for y in range(0, image.shape[1], step_size)]
    kp, des = brief.compute(image, kp)
    return np.float32(des)

def pyramid_dense_sift(image, step_size=5, num_level=2):
    # construct a SIFT object 
    sift = cv2.SIFT_create()
    all_des = []
    for l in range(num_level):
        # create dense keypoints and compute descriptors
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size) 
                                            for y in range(0, image.shape[1], step_size)]
        kp, des = sift.compute(image, kp)
        all_des.append(des)
        
        # resample image to the next level
        image = cv2.pyrDown(image)
    return np.vstack(all_des)

def spatial_pyramid(image, codebook, k, step_size=5, num_level=2):
    img_h, img_w = image.shape
    concat_his = []
    # for each level
    for l in range(num_level):
        num_grid = 2**l
        
        grid_size_w = img_w//num_grid
        grid_co_w = np.arange(0, img_w, grid_size_w) #coordinate of each grid
        
        grid_size_h = img_h//num_grid
        grid_co_h = np.arange(0, img_h, grid_size_h) #coordinate of each grid
        
        # loop through each grid
        for i in range(num_grid):
            for j in range(num_grid):
                # compute dense SIFT descriptors
                des = dense_sift(image[grid_co_h[j]:grid_co_h[j]+grid_size_h, 
                                       grid_co_w[i]:grid_co_w[i]+grid_size_w], step_size)
                
                # compute histogram of bovw
                his = histogram_bovw(des, codebook, k)
                his = his * (1/2**(num_level-l)) # weight
                concat_his.append(his)

    # concat histogram
    concat_his = np.array(concat_his).ravel()
    # normalizing
    concat_his = concat_his / concat_his.sum()
    return concat_his

def spatial_pyramid_orb(image, codebook, k, step_size=5, num_level=2):
    img_h, img_w = image.shape
    concat_his = []
    # for each level
    for l in range(num_level):
        num_grid = 2**l
        
        grid_size_w = img_w//num_grid
        grid_co_w = np.arange(0, img_w, grid_size_w) #coordinate of each grid
        
        grid_size_h = img_h//num_grid
        grid_co_h = np.arange(0, img_h, grid_size_h) #coordinate of each grid
        
        # loop through each grid
        for i in range(num_grid):
            for j in range(num_grid):
                # compute dense ORB descriptors
                des = dense_orb(image[grid_co_h[j]:grid_co_h[j]+grid_size_h, 
                                       grid_co_w[i]:grid_co_w[i]+grid_size_w], step_size)

                # compute histogram of bovw
                his = histogram_bovw(des, codebook, k)
                his = his * (1/2**(num_level-l)) # weight
                concat_his.append(his)

    # concat histogram
    concat_his = np.array(concat_his).ravel()
    # normalizing
    concat_his = concat_his / concat_his.sum()
    return concat_his

def spatial_pyramid_brief(image, codebook, k, step_size=5, num_level=2):
    img_h, img_w = image.shape
    concat_his = []
    # for each level
    for l in range(num_level):
        num_grid = 2**l
        
        grid_size_w = img_w//num_grid
        grid_co_w = np.arange(0, img_w, grid_size_w) #coordinate of each grid
        
        grid_size_h = img_h//num_grid
        grid_co_h = np.arange(0, img_h, grid_size_h) #coordinate of each grid
        
        # loop through each grid
        for i in range(num_grid):
            for j in range(num_grid):
                # compute dense BRIEF descriptors
                des = dense_brief(image[grid_co_h[j]:grid_co_h[j]+grid_size_h, 
                                       grid_co_w[i]:grid_co_w[i]+grid_size_w], step_size)

                # compute histogram of bovw
                his = histogram_bovw(des, codebook, k)
                his = his * (1/2**(num_level-l)) # weight
                concat_his.append(his)

    # concat histogram
    concat_his = np.array(concat_his).ravel()
    # normalizing
    concat_his = concat_his / concat_his.sum()
    return concat_his


def histogram_bovw(visual_words, codebook, k):
    his = np.zeros(k)
    for vw in visual_words:
        # find distance from vw to each representation vector (codebook)
        dist = np.power(np.power(np.tile(vw, (k, 1)) - codebook, 2).sum(axis=1), 0.5)
        # min distance
        min_codebook = dist.argsort()[0]
        # calculate histogram
        his[min_codebook] += 1
    return his
