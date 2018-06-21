## Udacity Self-Driving Car, Term1, 
### Project 4: **Vehicle Detection**

The goals / steps of this project are the following:

* Perform feature extraction using Histogram of Oriented Gradients (HOG), color features, and histogram of colors from a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run a pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### The rubric points are considered individually and it's implementation is describe below.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The following is a writeup of Project 4. Additionally, a README file containing the project summary is located in the main part of this repository. All code referenced in this write up can be found in the IPython notebook "vehicle_detection.ipynb"

### Histogram of Oriented Gradients (HOG), Color, and Histogram Features

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in IPython notebook under the section 'Feature Extraction Functions'. The majority of the code in this section was taken in whole, or adapted from, code provided in Lesson 20. 

HOG features were obtained using the function `get_hog_features()` which wraps the sklearn function `skimage.hog()`. HOG parameters were selected after training models using various features (HOG and others). The feature combination with the best accuracy was selected (see "Training Classifier" below). For HOG features this included using LUV color-space with all three channels an orientation = 9, pixels_per_cell = 16 and cells_per_block = 2 (Table 1). Figures 1 and 2 show training images and their respective hog features visualations.

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


For feature extraction the function `single_img_features()` wraps up the HOG, spatial, and histogram features functions. Parameters chosen for the additional features are listed in Table 2. The function `bin_spatial()` outputs spatial features by simply resizing the input YCrCb images to 16x16 pixels lists the values in an array. Histogram features were obtained using `color_hist()` to create histograms  of all three channels using a bin size of 32. The histograms were then concatenated together. Figures 3-5 shows the YCrCb, spatial features and histogram features of a sample test image.

[Table 1: Spatial, Histogram Features for Classifier]


| Feature Parameters | Values   | 
|:-------------:|:-------------:| 
| Color Space    | YCrCb     |
| Channel Number    | ALL      |
| Spatial (image) Size | 16 x 16     |
| Histogram Bin Size  | 32    | 


### Figure 3. Training Image in YCrCb Color Space.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/car_test_img_hog.png" height="480" width="640" />

### Figure 4. Plot of Spatial Freatures of Training Image.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/notcar_test_img_hog.png" height="480" width="640" />

### Figure 5. Plot of Training Image Histogram.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/notcar_test_img_hog.png" height="480" width="640" />



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training of the classifier can be found in section "Classifier Training" of the IPython notebook. Classification was done using a linear SVM. Test images were obtained from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark suit](http://www.cvlibs.net/datasets/kitti) (both links originally provided by Udacity). The training images were 64x64 pixels and presorted between vehicles and non-vehicles (for presorting code see python script 'getImages.py' in the repository). These totaled 8792 vehicle images and 8968 non-vehicles images. Additional images were added added to the vehicles training set from a Udacity provide [image set](http://bit.ly/udacity-annoations-crowdai), however multiple errors were encountered while parsing the "corrected" csv file, so only an additional 251 vehicle images where added. These vehicles images were extracted from a larger road scene and resized to 64x64 pixels. Finally, more images were extracted from preselected images from project_video.mp4. These included sections of the road, with no vehicles, and images of the white vehicle at a 'distance'. 

Training images were appended to two python lists: cars and notcars. The training images were then shuffled and a portion of the car images were randomly removed, so the number of vehicle and non-vehicle images were equal, using the `sklearn.utils.shuffle()`. The `n_samples` paramter was equal to the smallest length between the car and noncar list.

Next, the features for each training images was obtained by converting to YCrCb and appending the color space, spatial, histogram, and HOG features using the `extract_features()` (see section 'Feature Extraction Functions'). Testing features were extracted from 20% of the training features using `sklearn.model_selection.train_test_split()`. 

Then both training and testing data were normalized to unit variance `sklearn.preprocessing.StandardScaler().fit()` and `sklearn.preprocessing.StandardScaler().transform()`. Finally, the SVM model was training using `sklearn.svm.LinearSVC()`. The accuracy of the model tested using `sklearn.svm.LinearSVC().score()`. Results of the models uding various features parameters are listed in Table 2 above.  

### Table 2: Percent Accuracies of SVC models using various feature parameters.

<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/featureModelTesting.png" height="480" width="640" />


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding windows was implementaed using a fixed window size of 64x64 using 75% overlap. Along the y-axis the windows start at y = 360  and end at y = 660. Along the x-axis the entire width was covered. 

A 75% overlap was selected such that a given window could be positively classified multiple times. Through trial-and-error 75% 
privided a slightly more robust detection algorithm than smaller values (e.g. 50% or 25%). Using factors of four (e.g. 3/4 = 75%) work well mathematically given the 64x64 pixel images size. Figure 6 shows all sliding windows superimposed on an sample image. Figure 7 shows a subsampled image of a car with windows superimposed over it.


### Figure 6. Plot of Training Image Histogram.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/slidingWindows.jpg" height="480" width="640" />

### Figure 7. Plot of Training Image Histogram.
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Vehicle-Detection-and-Tracking/master/output_images/slideWindows_car2.jpg" height="480" width="640" />

During vehicle classification each window was subsampled and features were obtained using `single_img_features()` and run through the classifier, `clf.predict()`. If a window is classifed as a vehicle the corrdinates are stored in an array called `hot_windows`.  

To reduce false positives from during classification a heatmap was created, see section "Single Image Classification". First, the `hot_windows` array was passed to `add_heat()`, which takes the area of the positively classified window coordinates and adds "1" to a `np.zero_like()` image,for form a `heat`. The `heat` image was thresholded at a value of "3" using `apply_threshold()` to make the `heatmap` image. The "blobs " or positive areas remaining after thresholding in the heatmap were `scipy.ndimage.measurements.label()`. From the labels, bounding boxes were drawn on the original RGB image using `draw_label_bboxes()`

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


This process worked reasonable well for single test images but a more robust approach was needed to remove false positives from the video image. For this 10 `heat` images were stored in a circular buffer. After each frame the images in the buffer were summed and passed to `apply_threshold()` using a threshold value = 20. Then labels were found and bounding boxes drawn. 

Additionally, to track a vehicle's position over time a vehicle class was created to store information about it's location, bounding box size and area, and last frame since identified. 



###########################

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

