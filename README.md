# (Neurocomputing) A Deep Learning and Image Enhancement Based Pipeline for Infrared and Visible Image Fusion

This is a pytorch implementation for Infrared and Visible Image Fusion, python version 3.7  


Abstract: It is difficult to use supervised machine-learning methods for infrared (IR) and visible (VIS) image fusion (IVF) because of the shortage of ground-truth target fusion images, and image quality and contrast control are rarely considered in existing IVF methods. In this study, we proposed a simple IVF pipeline that converts the IVF problem into a supervised binary classification problem (sharp vs. blur) and uses image enhancement techniques to improve the image quality in three locations in the pipeline. We took a biological vision consistent assumption that the sharp region contains more useful information than the blurred region. A deep binary classifier based on a convolutional neural network (CNN) was designed to compare the sharpness of the infrared region and visible regions. The output score map of the deep classifier was treated as a weight map in the weighted average fusion rule. The proposed deep binary classifier was trained using natural visible images from the MS COCO dataset, rather than images from the IVF domain (called “cross domain training” here). Specifically, our proposed pipeline contains four stages: (1) enhancing the IR and VIS input images by linear transformation and the High-Dynamic-Range Compression (HDRC) method, respectively; (2) inputting the enhanced IR and VIS images to the trained CNN classifier to obtain the weight map; and (3) using a weight map to obtain the weighted average of the enhanced IR and VIS images; and (4) using single scale Retinex (SSR) to enhance the fused image to obtain the final enhanced fusion image. Extensive experimental results on public IVF datasets demonstrate the superior performance of our proposed approach over other state-of-the-art methods in terms of both subjective visual quality and 11 objective metrics. It was demonstrated that the complementary information between the infrared and visible images can be efficiently extracted using our proposed binary classifier, and the fused image quality is significantly improved. 

Note: we will provide the source code after the paper is accepted or published. Now you can click the link to run from stramlit app
https://eyob12-deploy-streamlit-app-for-deep-learning-model-test-64necx.streamlit.app/

The full paper: https://doi.org/10.1016/j.neucom.2024.127353

# Quick start
1. run 'Model_Test.py'
Input infrared, visible, enhanced infrared, and enhanced visible images to generate the fused image. To enhance images, execute the "main.m" code contained within the folder Enhancement. This folder contains enhanced images for simple.

Then using the code “Model_Test.py” generate the fused result

# Usage 
# Training 
Prepare dataset using the code “ create_training_dataset.m” from coco dataset according to the figure shown below. Then classify for training=1000000 micro patches, for test= 300,000 micro patches. 
Then using the code “Model_training.py” train the model  
# Preparation of clean and blur image pairs for training
![method of Inf and VIs](https://user-images.githubusercontent.com/57870274/184500375-8b786a3c-663f-43c4-9d9b-80f808be0d7d.jpg)
 The following is the steps of preparing the dataset, for further see in the above Fig
 1. Randomly selected image from COCO dataset is changed to grayscale
 2. Each randomly changed to grayscale image pass through four different Gaussian filters which the standard deviation is 9 and the cut of 9x9,11x11,13x13 and 15x15. Then, produce five different types of images, including the original image and four versions of the blurred version.
 3. The gradient in the direction of horizontal and vertical is applied for both the original and the four version of the blurred version.
 4. Create three groups that contain the original image, the gradient in the direction of horizontal G_x and the gradient in the direction of vertical G_y.

# Designing and training of blur detection neural network

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

# Existing infrared and visible image fusion methods
For the purpose of comparing the fusion performance of our proposed method to other state-of-the-art algorithms, 17 fusion methods are selected, which are Cross Bilateral Filter (CBF), Sparse Convolutional Representation (ConvSR), Structaware,  Discrete Cosine Harmonic Wavelet Transform (DCHWT), HybridMSD, DenseFuse, NestFuse, DDcGan,MEF-GAN, Image fusion network based on Proportional Maintenance of Gradient and Intensity (PMGI),CNN,MDlattLRR, Multi-layer Deep Features Fusion Method (VggML),Image Fusion Framework based on Convolutional Neural Network (IFCNN) (elementwise-maximum), Unified Unsupervised Image Fusion(U2Fusion), ResNet-Zca, and Residual Fusion Network for infrared and visible images(RFN-Nest) are adopted for comparation.

# Objective evaluation with the state-of-the-art methods
![paper_one evaluation](https://user-images.githubusercontent.com/57870274/216788219-fb1ae5cf-9b8a-42d2-98b3-03da3161cb4a.jpg)

The above table shows that our proposed method produces six best values(EI, SF, EN, MI, SD, and VIFF) and one second-best value (DF). It has the best value in EI and SF, which means that our proposed algorithm’s fusion image extracts more edge information and detailed information. The best value is also obtained in EN, which means that the fused image contains more information. In the MI indicators,  first, optimal values are obtained, indicating that the fused image sufficiently retains the information of the source images, and the fused image is more similar to the source images. It has the best value in  VIFF, which means that the image is more realistic or natural. The  optimal values of DF and second-best values SD indicate that our images  have high image contrast. The poor performance of $Q^{ab/f}$ and $N^{ab/f}$ suggests that our fusion image may have more noise. We believe that the reason for this result is the pixel level intensity different. Our method approach is based on the weight map generation, so the generated image is more dependent on the sours image's intensity level. Similar to adversarial networks, we do not generate the fused image with a generator, so the probability of noise occurrence increases, but we use blending techniques to reduce noise.

# Subjective evaluation with the state-of-the-art methods. 

![subjective_evaluation-1](https://user-images.githubusercontent.com/57870274/184528032-5fc5d61d-6154-449f-bcd2-75e5764b3729.jpg)

![subjective_evaluation-2](https://user-images.githubusercontent.com/57870274/184502502-80a824fb-ffd5-4d44-a326-2b0a6b033150.jpg)

![subjective_evaluation-3](https://user-images.githubusercontent.com/57870274/184502568-e0f5874f-9be9-4d29-8757-e05f0632460e.jpg)


