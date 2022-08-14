# Existing infrared and visible image fusion methods
For the purpose of comparing the fusion performance of our proposed method to other state-of-the-art algorithms, 17 fusion methods are selected, which are Cross Bilateral Filter (CBF), Sparse Convolutional Representation (ConvSR), Structaware,  Discrete Cosine Harmonic Wavelet Transform (DCHWT), HybridMSD, DenseFuse, NestFuse, DDcGan,MEF-GAN, Image fusion network based on Proportional Maintenance of Gradient and Intensity (PMGI),CNN,MDlattLRR, Multi-layer Deep Features Fusion Method (VggML),Image Fusion Framework based on Convolutional Neural Network (IFCNN) (elementwise-maximum), Unified Unsupervised Image Fusion(U2Fusion), ResNet-Zca, and Residual Fusion Network for infrared and visible images(RFN-Nest) are adopted for comparation.

# Comparation with the sate-of-the-art methods
![evaluation](https://user-images.githubusercontent.com/57870274/184528011-45302472-1ad0-4fbe-a09c-6bc37d7ecda2.png)
The above table shows that our proposed method produces six best values (EI, SF, EN, MI, SD, and VIFF) and one second-best value (DF). It has the best value in EI and SF, which means that our proposed algorithm’s fusion image extracts more edge information and detailed information. The best value is also obtained in EN, which means that the fused image contains more information. In the MI indicators,  first, optimal values are obtained, indicating that the fused image sufficiently retains the information of the source images, and the fused image is more similar to the source images. It has the best value in  VIFF, which means that the image is more realistic or natural. The  optimal values of DF and second-best values SD indicate that our images  have high image contrast. The poor performance of $Q^{ab/f}$ and $N^{ab/f}$ suggests that our fusion image may have more noise. We believe that the reason for this result is the pixel level intensity different. Our method approach is based on the weight map generation, so the generated image is more dependent on the sours image's intensity level. Similar to adversarial networks, we do not generate the fused image with a generator, so the probability of noise occurrence increases, but we use blending techniques to reduce noise.

# subjective evaluations 

![subjective_evaluation-1](https://user-images.githubusercontent.com/57870274/184528032-5fc5d61d-6154-449f-bcd2-75e5764b3729.jpg)

![subjective_evaluation-2](https://user-images.githubusercontent.com/57870274/184502502-80a824fb-ffd5-4d44-a326-2b0a6b033150.jpg)

![subjective_evaluation-3](https://user-images.githubusercontent.com/57870274/184502568-e0f5874f-9be9-4d29-8757-e05f0632460e.jpg)
