# Scene-Recognition

Scene recognition in computer vision is a challenge that involves labeling an image by analyzing its extracted features. This research article concentrates on three distinct approaches to scene recognition. The performance of each classifier will be assessed and compared on both the training and validation sets to determine their accuracy.

Types of scenes are: bedroom, Coast , Forest, Highway, industrial, Insidecity, kitchen, livingroom, Mountain, Office, OpenCountry, store, Street, Suburb, TallBuilding


Run1 - This run aims to develop a basic k-nearest-neighbor classifier utilizing the "tiny image" feature, which involves cropping each image to a square around its center and resizing it to a fixed resolution (preferably 16x16). The pixel values are then concatenated row by row to create a feature vector, with potential performance enhancement achieved by ensuring the tiny image has zero mean and unit length.

Run2 - In this run an ensemble of 15 one-vs-all linear classifiers using a bag-of-visual-words feature is built, which relies on fixed size densely-sampled pixel patches. To begin, it uses 8x8 patches of each image, sampled every 4 pixels in both the x and y directions. A subset of these patches is used to make a cluster using K-Means to create a vocabulary, (around 500 clusters). For each image, histogram is created using the vocabulary and then the same is used to create classifiers. Additionally, mean-centering and normalization of each patch before clustering and quantization is considered.

Run3 - In this run he images were processed to identify key points and extract feature descriptors. By applying the Bag of Visual Words (BoVW) approach, the images were converted into feature vectors. These feature vectors were then utilized as input for various Machine Learning Algorithms to perform scene recognition. 
1. Feature discriptors tried are: dense_sift, pyramid_dense_sift, spatial_pyramid, dense_orb, spatial_pyramid_orb , dense_brief, spatial_pyramid_brief
2. ML algorithms tried are: SVM, Naive Bayes and Random Forest,
