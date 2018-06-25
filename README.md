## Udacity Self-Driving Car, Term1, 
### Project 4: **Vehicle Detection**


Repository Contents include: 
* README.md - project summary
* writeup.md - project implementation
* vehicle_detection.ipynb - python code
* getImages.py - python script to extract training videos from labeled data set
* project_video_DetectT40.mp4 - video of vehicle dectection using linear SVM
* project_video_Detect_rbf.mp4 - video of vehicle dectection using SVM with rbf kernel

The goals / steps of this project are the following:

* Perform feature extraction using Histogram of Oriented Gradients (HOG), color features, and histogram of colors from a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run a pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


### Histogram of Oriented Gradients (HOG), Color, and Histogram Features

The project used HOG features, spatial features, and histogram features to classify vehicles in test images and in a video. Images were converted to YCrCb color space and features found using the parameters listed in Tables 1 and 2. 

[Table 1: HOG Parameters for Classifier]


| HOG Parameters | Values   | 
|:-------------:|:-------------:| 
| Color Space    | YCrCb     |
| hog_channel    | ALL     |
| orientation   | 9      | 
| pix_per_cell   | 16     |
| cell_per_block | 4     |

[Table 2: Spatial, Histogram Features for Classifier]


| Feature Parameters | Values   | 
|:-------------:|:-------------:| 
| Color Space    | YCrCb     |
| Channel Number    | ALL      |
| Spatial (image) Size | 16 x 16     |
| Histogram Bin Size  | 32    | 


### Figure 1. Car images and hog features.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/car_test_img_hog.png" height="480" width="640" />

### Figure 2. Car images and hog features.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/notcar_test_img_hog.png" height="480" width="640" />



### Figure 3. Training Image in YCrCb Color Space.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/Train_ycrcb.png" height="480" width="640" />

### Figure 4. Plot of Spatial Freatures of Training Image.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/bin_features.png" height="480" width="640" />

### Figure 5. Plot of Training Image Histogram.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/histogram.png" height="480" width="640" />



#### 3. Classifier Training

Test images were obtained from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark suit](http://www.cvlibs.net/datasets/kitti) (both links originally provided by Udacity). The training images were 64x64 pixels and presorted between vehicles and non-vehicles (for presorting code see python script 'getImages.py' in the repository).  Additional images were added to the vehicles training set from a Udacity provide [image set](http://bit.ly/udacity-annoations-crowdai), however multiple errors were encountered while parsing the "corrected" csv file, so only a fraction of the available images (251) where added. Finally, additional images were extracted from several frames in `project_video.mp4`. These included sections of the road and median barrier and images of the white vehicle at a distance. 

Training images were appended to two python lists: cars and notcars. The training images were then shuffled and the total length of each list was set equal to one another using `sklearn.utils.shuffle()`, with the `n_samples` paramter equal to the smallest of the two lengths: car or noncar.

Next, the features for each training image was obtained by converting each training image to YCrCb, finding the spatial, histogram, and HOG features with function`extract_features()` and appending the features together (see section 'Feature Extraction Functions'). 

A test set was extracted from 20% of the training set using `sklearn.model_selection.train_test_split()`. Both training and testing sets were normalized to unit variance with `sklearn.preprocessing.StandardScaler().fit()` and `sklearn.preprocessing.StandardScaler().transform()`. Finally, the SVM model was trained using `sklearn.svm.LinearSVC()`. The accuracy of the model tested using `sklearn.svm.LinearSVC().score()`. Results of different trained models, using various feature parameters are listed in Table 2 below.  

### Table 2: Percent Accuracies of SVC models using various feature parameters.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/featureModelTesting.png" height="480" width="640" />


### Sliding Window Search

A sliding windows algorithm was implemented using a fixed window size of 64x64 and an overlap of 75%. Along the y-axis the windows start and stop values were 360 and 660 respectfully. Along the x-axis the entire width of the image was covered. Figure 6 shows all sliding windows superimposed on an sample image. Figure 7 shows a subsampled image of a car with windows superimposed over it.


### Figure 6. Test Image with All Sliding Windows Superimpose.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/slidingWindows.jpg" height="480" width="640" />

### Figure 7. Car, Subsample of Test Image with Sliding Window.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/slideWindow_car2.png" height="480" width="640" />

During vehicle classification, each section of the image covered by a sliding window was subsampled and its features were obtained. Features were run through the classifier. 

### Single Image Classification
To reduce false positives from during classification a heatmap was created and thresholded with a value equal to 3. The positive areas remaining after thresholding were labeled and bounding boxes were drawn on the original RGB image.

### Figure 7. Vehicle Detection of Test Image.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/single_image.jpg" height="240" width="320" />


### Video Implementation


<a href="https://youtu.be/l6uH7AAM0e4" target="_blank"><img src="https://i.ytimg.com/vi/l6uH7AAM0e4/2.jpg?time=1529895207512" alt="Vehicle Detection Video Using Linear SVM" width="240" height="180" border="10" /></a>


The single image process worked reasonable well for the test images, but a more robust approach was needed to remove false positives from the project video. For this, 10 `heat` images were stored in a circular buffer. After each video frame the images in the buffer were summed and threshold at a value equal to 40. Finally, labels were found and bounding boxes drawn. 

This approached was sufficient but still yielded a few false positives. Additionally, between frames 600 and 800 the white car in the video was periodically undetected. A link to this video is provided below

After successive attempts to reduce false positives completely while identifying continuously identifying both cars in the video, a non-linear SVM was implemented using an RBF kernel. The C value was testing at 0.5, however it was found that the default value (C=1.0) worked the best. 

The circular buffer heatmap implementation described above was implemented to filter false positives, using a threshold value = 35. The RBF kernel implementation work very well, no false positives can be seen and the two cars on the right side of the image are correctly classified of the entire video.  However, processing the video was excrutiatingly long, and averages approximately 14 seconds per frame. A link to the processed video is listed below.

<a href="https://youtu.be/OW1txgFD_0o"><img src="https://i.ytimg.com/vi/OW1txgFD_0o/3.jpg?time=1529895060356" alt="Vehicle Detection Video Using Non-Linear SVM" width="240" height="180" border="10" /></a>

### Discussion

The biggest problems in the implementation of the video were 1) reduction of false positives and 2) finding a threshold value that would yield bounding boxes large enough to reasonably fit the detected vehicles yet successfully filter false positives. Single heat image tresholding used to process a single test image was insufficient for video processing.  Summing multiple heat images worked much better, but did not reduce all false positives. Increasing the histogram bin size from 16 to 32 also helped reduce false positives, however this had the unintended consequence of creating new false positives and increased the video processing time. 

The heat buffer was tested at a size of 5, 15, and 20. The larger buffer sized increased the blob values close to 255 and did not improve filtering perfomance. In the end it was challenging to find a balance between the total number of heat images in the circular buffer and a threshold value. 

The use of all 3 image channels, finding the HOG features, as well as the other fetures, of each sliding window made processing the video image frame slow: approximately 7.92 seconds per frame. Not adequate for real-time applications.

Using a Non-linear SVM with RBF kernelperformed much better at correctely classifying vehicles, but the frame processing time, in its current form, was much too long and not usable in a real-time applicaton. 
