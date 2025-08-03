# Lesson 14: Optimizing Deep Learning Models for Computer Vision

This repository contains a Jupyter Notebook titled **"Lesson 14 - Optimizing Deep Learning Models for Computer Vision"**, which provides practical techniques for making deep learning models more efficient — both in training and deployment.

## 🧠 Lesson Overview

Are you making your model design run efficiently? If so — are you optimizing for:

- 🚅 Training speed?
- ⚡ Inference speed?
- 🧠 Parameter efficiency?
- 💾 VRAM efficiency?

Is your **training data pipeline optimized**? What about your **deployment model**?

This lesson is designed as a **checklist** of best practices and performance tricks to make sure your models aren’t using more compute than needed.

### 🔥 Real Gains Possible

With just a few tweaks in model architecture and training loops, you can significantly reduce resource usage:

| Model Version                  | Parameters | Accuracy | Train Time/Epoch | Inference Time |
|-------------------------------|------------|----------|------------------|----------------|
| Naive                         | 7,476,742  | 82%      | 171s             | 52 ms/step     |
| + Pooling                     | 5,873,478  | 65%      | 164s             | 51 ms/step     |
| + Separable Convolutions      | 679,137    | 74%      | 250s             | 36 ms/step     |
| + JIT Compilation             | 679,137    | 76%      | 89s              | 19 ms/step     |
| + AMP (Mixed Precision)       | 679,137    | 74%      | 67s              | 22 ms/step     |
| + Fusing Layers               | 679,137    | 79%      | 66s              | 22 ms/step     |

### ⚙️ Key Techniques Covered

- ✅ **XLA + JIT Compilation**  
- ✅ **Layer Fusing**
- ✅ **AMP (Automatic Mixed Precision)**
- ✅ **Depthwise Separable Convolutions**
- ✅ **Pooling Instead of Flattening**
- ✅ **Efficient Data Pipelines** using `tf.data` API
- ✅ **Model Compression & Deployment Optimization** using:
  - TensorFlow Pruning API
  - Weight Clustering
  - Post-training Quantization
  - Collaborative Optimization for 10x smaller models

