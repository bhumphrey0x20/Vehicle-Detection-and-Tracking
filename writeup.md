## Udacity Self-Driving Car, Term1, 
### Project 4: **Vehicle Detection**

The goals / steps of this project are the following:

* Perform feature extraction using Histogram of Oriented Gradients (HOG), color features, and histogram of colors from a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run a pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### The rubric points are considered individually and it's implementation is describe below.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

The following is a writeup of Project 4. Additionally, a README file containing the project summary is located in the main part of this repository. All code referenced in this write up can be found in the IPython notebook "vehicle_detection.ipynb"

### Histogram of Oriented Gradients (HOG), Color, and Histogram Features

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in IPython notebook 'vehicle-detection.ipynb' under the section 'Feature Extraction Functions'. The majority of the code in this section was taken in whole, or adapted from, code provided in Lesson 20. 

HOG features were obtained using the function `get_hog_features()` which wraps the sklearn function `skimage.hog()`. HOG parameters were selected after training models using various features (HOG and others). The feature combination with the best accuracy ( LUV, All channels,etc) was selected however the color space was changed to YCrCb, as it appeared to remove  more false while testing on section of the project video (see "Training Classifier" below). HOG features are listed in Table 1. Figures 1 and 2 show training images and their respective hog features visualations.

[Table 1: HOG Parameters for Classifier]


| HOG Parameters | Values   | 
|:-------------:|:-------------:| 
| Color Space    | YCrCb     |
| hog_channel    | ALL     |
| orientation   | 9      | 
| pix_per_cell   | 16     |
| cell_per_block | 4     |




### Figure 1. Car images and hog features.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/car_test_img_hog.png" height="480" width="640" />

### Figure 2. Car images and hog features.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/notcar_test_img_hog.png" height="480" width="640" />


For feature extraction the function `single_img_features()` wraps up the HOG, spatial, and histogram feature functions. Parameters chosen for the additional features are listed in Table 2. The function `bin_spatial()` outputs spatial features by simply resizing the input YCrCb images to 16x16 pixels then converts the image into a single vector array. Histogram features were obtained using `color_hist()` using a bin size of 32 (original training was performed using 16 bins, however during video processing this was changed to 32 to reduce false positives). Figures 3-5 shows the YCrCb test sample image and its respective spatial features and histogram features plots.

[Table 2: Spatial, Histogram Features for Classifier]


| Feature Parameters | Values   | 
|:-------------:|:-------------:| 
| Color Space    | YCrCb     |
| Channel Number    | ALL      |
| Spatial (image) Size | 16 x 16     |
| Histogram Bin Size  | 32    | 



### Figure 3. Training Image in YCrCb Color Space.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/Train_ycrcb.png" height="480" width="640" />

### Figure 4. Plot of Spatial Freatures of Training Image.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/bin_features.png" height="480" width="640" />

### Figure 5. Plot of Training Image Histogram.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/histogram.png" height="480" width="640" />



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training of the classifier can be found in section "Classifier Training" of the IPython notebook. Initial classification was done using a linear SVM. Test images were obtained from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark suit](http://www.cvlibs.net/datasets/kitti) (both links originally provided by Udacity). The training images were 64x64 pixels and presorted between vehicles and non-vehicles (for presorting code see python script 'getImages.py' in the repository).  Additional images were added to the vehicles training set from a Udacity provide [image set](http://bit.ly/udacity-annoations-crowdai), however multiple errors were encountered while parsing the "corrected" csv file, so only a fraction of the available images (251) where added. Finally, additional images were extracted from several frames in `project_video.mp4`. These included sections of the road and median barrier and images of the white vehicle at a distance. 

Training images were appended to two python lists: cars and notcars. The training images were then shuffled and the total length of each list was set equal to one another using `sklearn.utils.shuffle()`, with the `n_samples` paramter equal to the smallest of the two lengths: car or noncar.

Next, the features for each training image was obtained by converting each training image to YCrCb, finding the spatial, histogram, and HOG features with function`extract_features()` and appending the features together (see section 'Feature Extraction Functions'). 

A test set was extracted from 20% of the training set using `sklearn.model_selection.train_test_split()`. Both training and testing sets were normalized to unit variance with `sklearn.preprocessing.StandardScaler().fit()` and `sklearn.preprocessing.StandardScaler().transform()`. Finally, the SVM model was trained using `sklearn.svm.LinearSVC()`. The accuracy of the model tested using `sklearn.svm.LinearSVC().score()`. Results of different trained models, using various feature parameters are listed in Table 2 below.  

### Table 2: Percent Accuracies of SVC models using various feature parameters.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/featureModelTesting.png" height="480" width="640" />


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A sliding windows algorithm was implemented using a fixed window size of 64x64 and an overlap of 75% (*note: overlap in the code = 0.25 with is the fraction of the window step, yeilding 75 overlap for each window). This was done through trial-and-error using overlaps of 75%, 50% and 25%. However, using a 75% overlap increased the coverage of windows over a vehicle and increased value of correctly classified blobs in the heatmap and for proper filtering of lower valued false positive blobs (discussed below). 

Sliding windows were created using `slide_window()` found in the "Feature Extraction Functions" section of the IPython notebook. Along the y-axis the windows start and stop values were 360 and 660 respectfully. Along the x-axis the entire width of the image was covered. Figure 6 shows all sliding windows superimposed on an sample image. Figure 7 shows a subsampled image of a car with windows superimposed over it.


### Figure 6. Test Image with All Sliding Windows Superimpose.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/slidingWindows.jpg" height="480" width="640" />

### Figure 7. Car, Subsample of Test Image with Sliding Window.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/slideWindow_car2.png" height="480" width="640" />

During vehicle classification, `search_windows()` was used to search each section of the image covered by a sliding window. The section was subsampled and its features were obtained using `single_img_features()`. Then those features were run through the classifier `sklearn.svm.SVC().predict()`. If a window was classifed as a vehicle, the coordinates were stored in a list and returned to the variable list `hot_windows`. See "Feature Extraction Functions" section of the IPython notebook.

To reduce false positives from during classification a heatmap was created (see section "Single Image Classification" in the IPython notebook). To create the heatmap the `hot_windows` list was passed to `add_heat()`. The fuction used the corrdinates of positively classified windows and added "1" to the corresponding area in a blank image. After all the `hot_windows` coordinates were processed. To remove potential false positives, the image was thresholded with a value equal to 3 using `apply_threshold()`. This created the `heatmap`. The positive areas remaining after thresholding were labeled using `scipy.ndimage.measurements.label()`. From the labels, bounding boxes were drawn on the original RGB image using `draw_label_bboxes()`. Heatmap and bounding box related functions may be found in the section "Heatmap Functions".  

### Figure 7. Vehicle Detection of Test Image.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/single_image.jpg" height="240" width="320" />


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)


<a href="https://youtu.be/l6uH7AAM0e4" target="_blank"><img src=https://i.ytimg.com/vi/l6uH7AAM0e4/2.jpg?time=1529894186711" alt="Vehicle Detection Test Video" width="240" height="180" border="10" /></a>


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The single image process worked reasonable well for the test images, but a more robust approach was needed to remove false positives from the project video. For this, 10 `heat` images were stored in a circular buffer. After each video frame the images in the buffer were summed, then a threshold = 35 was applied to filter the false positives, using `apply_threshold()`  (see section "Heatmap Functions"). Finally, labels were found and bounding boxes drawn. 

This approached was sufficient but still yielded a few false positives. Additionally, between frames 600 and 800 the white car in the video periodically lost it's bounding boxes(e.g. threshold is filtering out the car).

<a href="https://youtu.be/l6uH7AAM0e4" target="_blank"><img src="https://i.ytimg.com/vi/l6uH7AAM0e4/2.jpg?time=1529895207512" alt="Vehicle Detection Video Using Linear SVM" width="240" height="180" border="10" /></a>


#### 3. Another Approach: Non-Linear SVM

After successive attempts to reduce false positives completely while bounding boxes around both cars in the video a non-linear SVM was implemented. A RBF kernel was selected with a C-value = 0.5. The heatmap implementation described above was implemented to filter false positives and applied to the project video. The RBF kernel implementation work very well, however processing the video was excrutiatingly long. A link to the processed video is listed below.

<a href="https://youtu.be/OW1txgFD_0o"><img src="https://i.ytimg.com/vi/OW1txgFD_0o/3.jpg?time=1529895060356" alt="Vehicle Detection Video Using Non-Linear SVM" width="240" height="180" border="10" /></a>



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problems in the implementation of the video were 1) reduction of false positives and 2) finding a threshold value that would yield bounding boxes large enough to reasonably fit the detected vehicles. The original video implementation included tresholding a single heat image, but would not remove false positives. Summing multiple heat images worked much better but did not reduce all false positives. Increasing the histogram bin size from 16 to 32 also helped reduce false positives, but increased the video processing time, and in one instance createed a new false positve that did not exit with the smaller bin size. In the end it was challenging to find a balance between the total number of heat images in the circular buffer and a threshold value. 

The use of all 3 image channels, finding the HOG features, as well as the other fetures, of each sliding window made processing the video image frame slow: approximately 7.92 seconds per frame. Not adequate for real-time applications.

Using a Non-linear SVM with RBF kernel and C= 0.5 performed much better at correctely classifying vehicles, but the frame processing time, in its current form, was much too long and not usable in a real-time applicaton. 
