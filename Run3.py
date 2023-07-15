#Importing Necessary packages

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, top_k_accuracy_score
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV





"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## Initiate varibles"""

k = 200 # number of representative vectors
feature_type = 'spatial_pyramid' 
# ['dense_sift', 'pyramid_dense_sift', 'spatial_pyramid', 'dense_orb', 'spatial_pyramid_orb' ,'dense_brief', 'spatial_pyramid_brief'] List to choose the type of feature set from
sift_step_size = 5
num_level = 2


"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 1. Dataset preparation"""

data = {'img_path': [], 'label': []} #Storing path of images and labels in a dictionary

for root, dirs, files in os.walk("./training/"):
    label = os.path.basename(root)
    for file in files:
        if file.endswith('.jpg'):
            data['img_path'].append(os.path.join(root, file))
            data['label'].append(label)

df_data = pd.DataFrame(data)




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 2. Feature extraction"""

# Function to extract sift feature discriptors from image 
def sift(image, step_size=5):
    # construct a SIFT object
    sift = cv2.SIFT_create()
    # find keypoints and descriptors
    kp, des = sift.detectAndCompute(image, None)
    return des

# Function to extract dese sift feature discriptors from image 
def dense_sift(image, step_size=5):
    # construct a SIFT object
    sift = cv2.SIFT_create()
    # create dense keypoints and compute descriptors
    kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size)
                                        for y in range(0, image.shape[1], step_size)]
    kp, des = sift.compute(image, kp)
    return des

# Function to extract dense orb feature discriptors from image 
def dense_orb(image, step_size=5):
    # read image from img_path then convert to gray scale
    # construct a ORB object
    orb = cv2.ORB_create()
    # create dense keypoints and compute descriptors
    kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size)
                                        for y in range(0, image.shape[1], step_size)]
    kp, des = orb.compute(image, kp)
    return np.float32(des)

# Function to extract dense brief feature discriptors from image 
def dense_brief(image, step_size=5):
    # construct a BRIEF object
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # create dense keypoints and compute descriptors
    kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, image.shape[0], step_size)
                                        for y in range(0, image.shape[1], step_size)]
    kp, des = brief.compute(image, kp)
    return np.float32(des)

# Function to calculate pyrmid_dense_sift feature discriptors from image
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

# Function to calculate spatial_pyrmid_dense_sift feature discriptors from image and return the histogram from bag of visual words
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

# Function to calculate spatial_pyrmid_orb feature discriptors from image and return the histogram from bag of visual words for the image
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

# Function to calculate spatial_pyrmid_brief feature discriptors from image and return the histogram from bag of visual words from the image
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




visual_words = [] #List to store feature discriptors 

#Iterating over each images in the dataset and idetifying feature points
for i, row in df_data.iterrows():
    # read image from img_path then convert to gray scale
    image = cv2.imread(row['img_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get descriptors for each image
    if feature_type in ['dense_sift', 'pyramid_dense_sift', 'spatial_pyramid']:
        visual_words.append(dense_sift(image, sift_step_size))
    elif feature_type in ['dense_orb', 'spatial_pyramid_orb']:
        visual_words.append(dense_orb(image, sift_step_size))
    elif feature_type in ['dense_brief', 'spatial_pyramid_brief']:
        visual_words.append(dense_brief(image, sift_step_size))
    elif feature_type == 'dense_surf':
        visual_words.append(dense_surf(image, sift_step_size))

df_data['visual_words'] = visual_words





"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 4. Learning Vocabulary"""

# preparing bag of visual words
BoVW = df_data['visual_words'].to_list()
BoVW = np.vstack(BoVW)

# K-Means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centres = cv2.kmeans(BoVW, k, None, criteria, 10, flags)


"""### Histograms of bags of visual words"""
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

#Creating histogram for each image using feature points
his_bovw = []
print(feature_type)

#iterating over images
for i, row in df_data.iterrows():
    # read image from img_path then convert to gray scale
    image = cv2.imread(row['img_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dense SIFT
    if feature_type == 'dense_sift':
        his_bovw.append(histogram_bovw(dense_sift(image, sift_step_size),
                                       centres, k))

    # pyramid dense SIFT
    elif feature_type == 'pyramid_dense_sift':
        his_bovw.append(histogram_bovw(pyramid_dense_sift(image, sift_step_size,
                                                          num_level), centres, k))
    # dense SIFT with spatial pooling
    elif feature_type == 'spatial_pyramid':
        his_bovw.append(spatial_pyramid(image, centres, k, sift_step_size, num_level))

    # dense ORB
    elif feature_type == 'dense_orb':
        his_bovw.append(histogram_bovw(dense_orb(image, sift_step_size),
                                       centres, k))

    # dense ORB with spatial pooling
    elif feature_type == 'spatial_pyramid_orb':
        his_bovw.append(spatial_pyramid_orb(image, centres, k, sift_step_size, num_level))

    # dense BRIEF
    elif feature_type == 'dense_brief':
        his_bovw.append(histogram_bovw(dense_brief(image, sift_step_size),
                                       centres, k))

    # dense BRIEF with spatial pooling
    elif feature_type == 'spatial_pyramid_brief':
        his_bovw.append(spatial_pyramid_brief(image, centres, k, sift_step_size, num_level))

his_bovw = np.array(his_bovw)




"""---------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 4. Classifier"""


"""## Preparing Dataset"""
X = his_bovw
y = np.array(df_data['label'].to_list())

#Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

# normalizing the images
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)


"""### SVC (kernel: rbf)"""

clf = SVC(probability=True).fit(X_train_norm, y_train)
y_pred = clf.predict(X_val_norm)
print('accuracy SVM:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
confusion_matrix(y_val, y_pred)


"""### Multinomial Naive Bayes"""

# clf = MultinomialNB().fit(X_train, y_train)
clf = GaussianNB().fit(X_train, y_train)
# clf = BernoulliNB().fit(X_train, y_train)

y_pred = clf.predict(X_val)
print('accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))



"""### Random forest"""

clf = RandomForestClassifier().fit(X_train_norm, y_train)

y_pred = clf.predict(X_val_norm)
print('accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# top k accuracy
top_k_accuracy_score(y_val, clf.predict_proba(X_val_norm), k=2)



"""### SVM with Hyperparameter tuning"""

parameters = {'kernel':['rbf'], 'C':np.linspace(1, 10, 50), 'gamma': [0.1, 1.0, 10, 100]}

grid_search = GridSearchCV(SVC(), parameters)
grid_search.fit(X_train, y_train)

grid_search.best_score_

grid_search.best_params_

y_pred = grid_search.predict(X_val)
print('accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))




"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 5. Making predicitions on test set usig the best classifier i.e Spatial Pyramid Dense SIFT"""

test_data = [] #List to store features of images
test_file_name = [] #list to store the path of images

#Extracting feature from test data
for root, dirs, files in os.walk("./testing/"):
    for file in files:
        if file.endswith('.jpg'):
            test_file_name.append(file)

            img_path = os.path.join(root, file)
            # read image from img_path then convert to gray scale
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            test_data.append(spatial_pyramid(image, centres, k, sift_step_size, num_level))

# Creating training set and test set
X_train = X.copy()
X_test = np.array(test_data)
y_train = y.copy()

# normalizing the train and test set
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

#Creating a classifier
clf = SVC(probability=True, C=2.2).fit(X_train_norm, y_train) 

y_pred = clf.predict(X_test_norm) # prediction on the test data

#Storing predictions in a datframe for testing dataset
lists = [m + " " + n for m, n in zip(test_file_name, y_pred)]
re = pd.DataFrame(lists, columns=['Name'])
re['Num'] = [int(x.split('.')[0]) for x in re['Name']]
re = re.sort_values(by=['Num'])
re = re.reset_index()
sub = list(re['Name'])

#Saving the predictions in a txtfile
with open(r'run3.txt', 'w') as fp:
    for item in sub:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')