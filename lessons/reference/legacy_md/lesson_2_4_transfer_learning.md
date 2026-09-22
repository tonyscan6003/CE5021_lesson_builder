# lesson_2_4_transfer_learning

Deep Learning for Computer vision 

Transfer Learning Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Deep Learning 

- Success of Deep Learning 

   - Large Datasets [1] 

   - Deep Neural Network Architectures 

   - High Performance Computing 

- Deploying Models to new Domains? 

   - Insufficient Data in new Domain 

   - Computational Time/resources to train model (even if data is available) 

- Transfer Learning as a solution 

[1] Revisiting Unreasonable Effectiveness of Data in Deep Learning Era 

- (a) Confluence of Technologies behind Deep Learning 

- (b) Applications of Deep Learning in Machine Vision. 

©Dr. Tony Scanlan 

# Transfer Learning 

- Deep Neural Network can be partitioned in a “Feature Extractor” and classifier. 

- Feature Extractor learns to recognise many _generic_ image features in images. [2] • From Large Training Dataset 

- Feature Extractor may be deployed on different domain (TF Tutorial). • Small Training Set • Train Classifier • Fine Tune Feature Detector. 

[2] <u>Tan, C., Sun, F., Kong, T., Zhang, W., Yang, C. and Liu, C., 2018, October. A survey on deep transfer learning. In</u> _<u>International conference on artificial neural networks</u>_ <u>(pp. 270-279). Springer, Cham.</u> 

©Dr. Tony Scanlan 

# Object Detection 

- Transfer Learning enables CNN based Object Detection 

- Pre-trained Convolution layers activate in response to common objects. 

- Spatial Information is contained in Deep Layers. 

- Bounding Box Predictor (& Class label) can be trained with supervised learning. • Annotated Data required. 

©Dr. Tony Scanlan 

# Should we always use Transfer Learning? 

- Using a pre-trained Network (Image Classifier on ImageNet) has become standard practice for vision tasks. 

   - Particularly when limited by Data & Computation 

- Observations from He _et al._ [4] 

   - Training networks from scratch possible (Don’t need specialised architectures) 

   - Convergence faster with pre-training 

   - Training from scratch no-worse than with pretraining (until target domain data is too small) 

   - Total training time is long when pre-training is considered 

Image reproduced from [4] 

[4] <u>He, K., Girshick, R. and Dollár, P., 2019. Rethinking imagenet pre-training. In</u> _<u>Proceedings of the IEEE/CVF International Conference on Computer Vision</u>_ <u>(pp. 4918-4927).</u> 

©Dr. Tony Scanlan 

# Transfer Learning and Specialised Tasks? 

- Transfer learning may not work as well when there is a big difference between images in target domain and data used for pre-training (Raghu _et al_ . [5]). 

- Randomly initialised & lightweight (few parameters) models may perform as 

- well as a transfer learning model. 

- Transfer learning helps larger models with more parameters when amount of data is target domain is very small. 

Table showing Performance of various models trained from random initialisation or via Transfer learning on Diabetic Retinopathy dataset. 

- [5] Raghu, M., Zhang, C., Kleinberg, J. and Bengio, S., 2019. Transfusion: Understanding transfer learning for <u>medical imaging.</u> _<u>arXiv preprint arXiv:1902.07208</u>_ <u>.</u> 

©Dr. Tony Scanlan 

# Transfer Learning Recommendations 

- Consideration needs to be given to source & target domains similarity, label similarity and size of datasets 

- • Similar (large) Source and Target Domain • Excellent Performance even without fine tuning • Better than training from scratch (Small target Dataset) • Equivalent performance when trained from scratch (large target dataset) 

- Less Similar (large) Source and Target (medium/small) Domain • Fine tuning will be necessary for accuracy • Risk overfitting (small dataset) 

- Dissimilar (Large) Source and Target Domain • Equivalent performance when trained from scratch (medium target dataset) • Slightly better than training from scratch (small target datasets) 

- Also consider computational resources available for training. • Pre-trained model may good solution with limited computation. • Saves training time! 

©Dr. Tony Scanlan 

# Pretrained Models 

- <u>Keras Applications</u> 

   - Standard pretrained image recognition models 

   - Resnet50, VGG16, DenseNet, mobilenet 

   - Can be used as a basis for other applications with Transfer Learning 

   - Can freeze Feature Extraction part or have Fine Tuning 

- Easily instantiated 

   - See transfer learning tutorial 

# Create the base model from the pre-trained model MobileNet V2 IMG_SHAPE = IMG_SIZE + (3,) 

base_model = 

tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, 

include_top=False, weights='imagenet') 

Note: include_top = False Removes Classification part of network Leaving feature extraction part
