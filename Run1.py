#Importing necessary packages
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 1. Dataset preparation"""

data = {'img_path': [], 'label': []} #Storing path of images and labels in a dictionary

for root, dirs, files in os.walk("./training/"):
    label = os.path.basename(root)
    for file in files:
        if file.endswith('.jpg'):
            data['img_path'].append(os.path.join(root, file))
            data['label'].append(label)

df_data = pd.DataFrame(data)




"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 2. Feature extraction"""

def feature_extraction(paths):
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

df_data['features'] = feature_extraction(df_data['img_path'])




"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 3. Data Split"""

X = np.array(df_data['features'].to_list())
y = np.array(df_data['label'].to_list())
print(X.shape, y.shape)

# split data into train and val set with 10% data in the val set.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 521)




"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 4. Classifier : k neighbour classifier """

# instantiate k neighbour classifier with 100 clusters
neigh = KNeighborsClassifier(n_neighbors=100)
# fit classifier on training data and train labels
neigh.fit(X_train, y_train)

# calculate accuracy of validation data
accuracy_score(y_val, neigh.predict(X_val))

# generate the classification report for validation set
y_pred = neigh.predict(X_val)
print('accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

confusion_matrix(y_val, y_pred) # Printing confusion matrix of validation set




"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""## 5. Making predicitions on test set"""


#Updating file name in test_file_name
test_file_name = []
for root, dirs, files in os.walk("./training/"):
    label = os.path.basename(root)
    for file in files:
        if file.endswith('.jpg'):
            test_file_name.append(os.path.join(root, file))

test_data = feature_extraction(test_file_name) #Extracting feature from test data


# prediction on the test data
y_pred = neigh.predict(test_data)

#Storing predictions in a datframe for testing dataset
lists = [m + " " + n for m, n in zip(test_file_name, y_pred)]
re = pd.DataFrame(lists, columns=['Name'])
re['Num'] = [int(x.split('.')[0]) for x in re['Name']]
re = re.sort_values(by=['Num'])
re = re.reset_index()
sub = list(re['Name'])

#Saving the predictions in a txtfile
with open(r'run1.txt', 'w') as fp:
    for item in sub:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')