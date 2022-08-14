# Deep_infrared_and_visible_image_fusion
 Cross Domain Deep Blur Simulation and Detection Network for Infrared and Visible Image Fusion

This is a pytorch implementation for Infrared and Visible Image Fusion, python version 3.7  


Abstract: Infrared and visible image fusion (IVIF)  is generally a challenging problem due to the unavailability of ground truth image. In this paper,  a fully automatic deep learning based IVIF method is proposed.  The IVIF problem is treated as a blur detection problem and an efficient multi-path convolution network is proposed for blur detection. A cross domain learning methodology is used to train the proposed network. We fused the enhanced images rather than original images.   Our proposed method mainly consists of  four  stages: 1)preparation of clean and blur image pairs for training; 2) designing and training of blur detection neural network; 3)enhancement of infrared and visible images; 4) fusing enhanced version of  infrared and visible image based on weighting map from trained blur detection network. Extensive experimental results in three public IVIF data sets demonstrate the superior performance of our proposed approach over other state-of-the-art methods in terms of both subjective visual quality and six objective metrics.

# Quick start
1. run 'Model_Test.py'
Input infrared, visible, enhanced infrared, and enhanced visible images to generate the fused image. To enhance images, execute the "main.m" code contained within the folder Enhancement. This folder contains enhanced images for simple.

Then using the code “Model_Test.py” generate the fused result

# Usage 
# Training 
Prepare dataset using the code “ create_training_dataset.m” from coco dataset according to the figure shown below. Then classify for training=1000000 micro patches, for test= 300,000 micro patches. 
Then using the code “Model_training.py” train the model  
#Preparation of clean and blur image pairs for training

![method of Inf and VIs](https://user-images.githubusercontent.com/57870274/184500375-8b786a3c-663f-43c4-9d9b-80f808be0d7d.jpg)
 The following is the steps of preparing the dataset, for further see in the above Fig
 1. Randomly selected image from COCO dataset is changed to grayscale
 2. Each randomly changed to grayscale image pass through four different Gaussian filters which the standard deviation is 9 and the cut of 9x9,11x11,13x13 and 15x15. Then, produce five different types of images, including the original image and four versions of the blurred version.
 3. The gradient in the direction of horizontal and vertical is applied for both the original and the four version of the blurred version.
 4. Create three groups that contain the original image, the gradient in the direction of horizontal G_x and the gradient in the direction of vertical G_y.

#Designing and training of blur detection neural network

![model](https://user-images.githubusercontent.com/57870274/184501173-4657b7e4-7981-45cb-8937-1b64d96cce56.jpg)

![ResNet_model](https://user-images.githubusercontent.com/57870274/184501286-6ee5e27e-fd19-4111-a01a-7c0cc9ef93c5.jpg)

# Test 
run 'Model_Test.py'  

# Evaluation 
Using both objective and subjective evaluation matrix evaluate the performance of the model 
For objective evaluation run the “main.m” code in the folder objective evaluation. 
The performance of the model is shown below for both objective and subjective ways.  

# The objective evaluation with the Threee Datasets: TNO, LLVIP, and VOT2020-RGBT 
![model output with dataset_evaluation](https://user-images.githubusercontent.com/57870274/184501831-2b77a684-d945-49aa-aae3-8204fe7e8a40.png)

# Subjective evaluation with the state-of-the-art methods.
![subjective_evaluation-1](https://user-images.githubusercontent.com/57870274/184502403-5d0bd8f8-ee17-46e8-ba48-84425f3eeda9.jpg)
![subjective_evaluation-2](https://user-images.githubusercontent.com/57870274/184502502-80a824fb-ffd5-4d44-a326-2b0a6b033150.jpg)

![subjective_evaluation-3](https://user-images.githubusercontent.com/57870274/184502568-e0f5874f-9be9-4d29-8757-e05f0632460e.jpg)

# Objective evaluation with the state-of-the-art methods

![evaluation](https://user-images.githubusercontent.com/57870274/184502710-f56b977b-3948-49fa-80cf-a7740cbdbf45.png)

The above table shows that our proposed method produces six best values (EI, SF, EN, MI, SD, and VIFF) and one second-best value (DF). It has the best value in EI and SF, which means that our proposed algorithm’s fusion image extracts more edge information and detailed information. The best value is also obtained in EN, which means that the fused image contains more information. In the MI indicators,  first, optimal values are obtained, indicating that the fused image sufficiently retains the information of the source images, and the fused image is more similar to the source images. It has the best value in  VIFF, which means that the image is more realistic or natural. The  optimal values of DF and second-best values SD indicate that our images  have high image contrast. The poor performance of $Q^{ab/f}$ and $N^{ab/f}$ suggests that our fusion image may have more noise. We believe that the reason for this result is the pixel level intensity different. Our method approach is based on the weight map generation, so the generated image is more dependent on the sours image's intensity level. Similar to adversarial networks, we do not generate the fused image with a generator, so the probability of noise occurrence increases, but we use blending techniques to reduce noise. 
