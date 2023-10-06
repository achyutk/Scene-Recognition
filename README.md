# Scene-Recognition

Scene recognition in computer vision is a challenge that involves labeling an image by analyzing its extracted features. This research article concentrates on three distinct approaches to scene recognition. The performance of each classifier will be assessed and compared on both the training and validation sets to determine their accuracy.

Types of scenes are: bedroom, Coast , Forest, Highway, industrial, Insidecity, kitchen, livingroom, Mountain, Office, OpenCountry, store, Street, Suburb, TallBuilding

![IMG](https://github.com/achyutk/Scene-Recognition/assets/73283117/b6bccb34-57b3-4e1f-adbc-0db305315480)


Run1 - This run aims to develop a basic k-nearest-neighbor classifier utilizing the "tiny image" feature, which involves cropping each image to a square around its center and resizing it to a fixed resolution (preferably 16x16). The pixel values are then concatenated row by row to create a feature vector, with potential performance enhancement achieved by ensuring the tiny image has zero mean and unit length.

Run2 - In this run an ensemble of 15 one-vs-all linear classifiers using a bag-of-visual-words feature is built, which relies on fixed size densely-sampled pixel patches. To begin, it uses 8x8 patches of each image, sampled every 4 pixels in both the x and y directions. A subset of these patches is used to make a cluster using K-Means to create a vocabulary, (around 500 clusters). For each image, histogram is created using the vocabulary and then the same is used to create classifiers. Additionally, mean-centering and normalization of each patch before clustering and quantization is considered.

Run3 - In this run he images were processed to identify key points and extract feature descriptors. By applying the Bag of Visual Words (BoVW) approach, the images were converted into feature vectors. These feature vectors were then utilized as input for various Machine Learning Algorithms to perform scene recognition. 
1. Feature discriptors tried are: dense_sift, pyramid_dense_sift, spatial_pyramid, dense_orb, spatial_pyramid_orb , dense_brief, spatial_pyramid_brief
2. ML algorithms tried are: SVM, Naive Bayes and Random Forest,

# Dataset

The dataset used for this proejct was provided by the University of Southampton. It can be accessed from here: https://drive.google.com/file/d/1hw0cbYrfuXnN54CI8-hWy6Fntm1j2jPL/view?usp=sharing


# Installations

> pip install opencv-python <br>
> pip install sklearn

Download the dataset and past it in the *data* folder. Note: Keep the folder structure as *./training/scenes/imgs*

> execute run1.ipynb file for Run1 implementation <br>
> execute run2.ipynb file for Run2 implementation <br>
> execute run3.ipynb file for Run3 implementation

# Results

Run1: Validation accuracy 20%> <br>

Run2: Validation accuaracy in different cluster sizes:<br>
| Cluster Size | Validation accuracy    |
| :---: | :---: |
| 250 | 48%   |
| 500 | 50%   |
| 600 | 44%   |


Run3:
The scores here are for Vocabulary size : 250
| Feature encoding  | SVM    | Naive Bayes| Random Forest    |
| :---: | :---: | :---: | :---: |
| Dense SIFT | 69.6%   | 61%   | 66.33%   |
| Dense SIFT with spatial pooling| 72.67%   | 63%   | 69.67%   |
| Dense ORB | 58%   | 53.33%   | 50.67%   |
| Dense ORB with spatial pooling | 59.33%   | 52%   | 61.33%   |
| Dense BRIEF | 54%   | 48%   | 49%   |
| Dense BRIEF with spatial pooling| 58.33%   | 49.33%   | 49.67%   |

Scores for vocabulary sizes 200 and 50 can be found in this paper: https://drive.google.com/file/d/1-OlGZNHIMpCdY7tyiYRvFTmR-OQKWQkz/view?usp=sharing
<!-- Vocabulary size : 200
| Feature encoding  | SVM    | Naive Bayes| Random Forest    |
| :---: | :---: | :---: | :---: |
| Dense SIFT | 71.33%   | 61%   | 68.67%   |
| Dense SIFT with spatial pooling| 73.33%   | 59.33%   | 68.67%   |
| Dense ORB | 55%   | 51.67%   | 53%   |
| Dense ORB with spatial pooling | 57.33%   | 51.33%   | 52.67%   |
| Dense BRIEF | 53%   | 47.33%   | 46.67%   |
| Dense BRIEF with spatial pooling| 58%   | 47%   | 50.67%   |


Vocabulary size : 50
| Feature encoding  | SVM    | Naive Bayes| Random Forest    |
| :---: | :---: | :---: | :---: |
| Dense SIFT | 68%   | 63.33%   | 65.33%   |
| Dense SIFT with spatial pooling| 71%   | 64.33%   | 67.67%   |
| Dense ORB | 55.67%   | 47%   | 51.67%   |
| Dense ORB with spatial pooling | 54.67%   | 51%   | 59.67%   |
| Dense BRIEF | 50%   | 42.67%   | 47%   |
| Dense BRIEF with spatial pooling| 54%   | 42.67%   | 51.33%   | -->





