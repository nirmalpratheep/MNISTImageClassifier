# MNIST Image Classifier: A Progressive Learning Journey

This project demonstrates the evolution of deep learning models for MNIST digit classification, showcasing three progressively sophisticated approaches that build upon each other's learnings. Each model represents a different stage in understanding CNN architecture, optimization techniques, and advanced training strategies.

## 🎯 Project Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each 28×28 pixels. This project implements three distinct CNN models that progressively improve in accuracy and sophistication:

- **Model 1**: Basic CNN setup targeting 95% accuracy with ~8K parameters
- **Model 2**: Enhanced with Batch Normalization and Global Average Pooling
- **Model 3**: Advanced model with data augmentation, optimized learning rate scheduling, and improved architecture targeting 98.4% accuracy

## 🏗️ Architecture Comparison

| Feature | Model 1 | Model 2 | Model 3 |
|---------|---------|---------|---------|
| **Total Parameters** | 7,616 | 7,798 | 7,762 |
| **Convolutional Layers** | 7 | 7 | 6 |
| **Batch Normalization** | ❌ | ✅ | ✅ |
| **Global Average Pooling** | ❌ | ✅ | ✅ |
| **Dropout** | ❌ | ✅ (0.1) | ✅ (0.0) |
| **Max Pooling** | 2 layers | 2 layers | 3 layers |
| **Data Augmentation** | Basic | Basic | Advanced |
| **Learning Rate Schedule** | StepLR(15, 0.1) | StepLR(15, 0.1) | OneCycleLR |
| **Final Accuracy** | 98.89% | 99.18% | 99.46% |

## 📊 Performance Analysis

### Model 1: Foundation Building
```
Target: 95% accuracy in 15 epochs with ~8K parameters
Achieved: 98.89% test accuracy
```

**Architecture Details:**
- **Input**: 28×28×1 grayscale images
- **Conv1**: 1→16 channels, 3×3 kernel
- **Conv2**: 16→8 channels, 3×3 kernel  
- **Conv3**: 8→16 channels, 3×3 kernel + MaxPool(2)
- **Conv4**: 16→8 channels, 3×3 kernel
- **Conv5**: 8→16 channels, 3×3 kernel + MaxPool(2)
- **Conv6**: 16→8 channels, 3×3 kernel, padding=1
- **Conv7**: 8→10 channels, 3×3 kernel, padding=1
- **FC1**: 90→10 features (flattened 10×3×3)

**Key Learnings:**
- Basic CNN structure with alternating channel patterns
- Two max pooling layers for spatial reduction
- Simple ReLU activations
- Traditional fully connected layer approach

### Model 2: Normalization & Pooling Enhancement
```
Target: Maintain ~8K parameters while adding BatchNorm and GAP
Achieved: 99.18% test accuracy
```

**Architecture Enhancements:**
- **Batch Normalization**: Added after every convolutional layer
- **Global Average Pooling**: Replaced FC layer with GAP + 10 output channels
- **Dropout**: Added 0.1 dropout for regularization
- **Channel Adjustment**: Modified conv6 to 16→12 channels for GAP compatibility

**Key Improvements:**
- **Faster Convergence**: BatchNorm enables higher learning rates
- **Better Generalization**: Dropout prevents overfitting
- **Parameter Efficiency**: GAP reduces parameters while maintaining performance
- **Stable Training**: Normalized activations lead to more stable gradients

### Model 3: Advanced Optimization
```
Target: 98.4% accuracy with data augmentation and optimized training
Achieved: 99.46% test accuracy
```

**Advanced Features:**
- **Data Augmentation**: RandomRotation(-7°, +7°) with fill=(1,)
- **Optimized LR Schedule**: OneCycleLR (cyclical LR up then down within one run)
- **Enhanced Architecture**: 3 max pooling layers for better spatial hierarchy
- **Strategic Channel Design**: 1→8→8→12→12→24→10 progression
- **Parameter Count**: 7,762 parameters (most efficient of all models)

**Training Optimizations:**
- **Custom Transforms**: Model-specific augmentation pipeline
- **Aggressive LR Decay**: Faster learning rate reduction
- **Multiple Pooling**: Better feature extraction at different scales

## 🔬 Detailed Analysis

### Why These Models Are Effective

#### 1. **Progressive Complexity**
Each model builds upon the previous one's learnings:
- Model 1 establishes the basic CNN foundation
- Model 2 introduces modern training techniques (BatchNorm, GAP)
- Model 3 demonstrates advanced optimization strategies

#### 2. **Parameter Efficiency**
All models maintain ~8K parameters, proving that:
- **Quality over Quantity**: Well-designed architectures outperform larger, poorly structured networks
- **Modern Techniques**: BatchNorm and GAP provide significant benefits with minimal parameter overhead
- **Strategic Design**: Channel progression and pooling placement matter more than raw parameter count

#### 3. **Training Stability**
- **Model 1**: Shows the importance of proper architecture design
- **Model 2**: Demonstrates how BatchNorm stabilizes training and enables faster convergence
- **Model 3**: Proves that data augmentation and learning rate scheduling are crucial for high performance

### Key Technical Insights

#### **Batch Normalization Impact**
- **Model 1 → Model 2**: +0.29% accuracy improvement
- Enables higher learning rates without instability
- Reduces internal covariate shift
- Acts as implicit regularization

#### **Global Average Pooling Benefits**
- Reduces parameter count (90 → 10 parameters in final layer)
- Prevents overfitting by eliminating fully connected layers
- Provides spatial invariance
- Maintains performance while improving efficiency

#### **Data Augmentation Strategy**
- **Model 3**: Custom rotation augmentation (-7° to +7°)
- Improves generalization to unseen variations
- Reduces overfitting on training data
- Essential for achieving >98% accuracy

#### **Learning Rate Scheduling**
- **Model 1 & 2 – StepLR**: Piecewise-constant LR with periodic drops (step_size, gamma). Simple, stable, and effective when plateaus are known.
- **Model 3 – OneCycleLR**: LR increases to a max then anneals to a very low value over the training run; typically momentum varies inversely. Leads to fast early learning, robust exploration, and fine-grained end fitting, often improving generalization and peak accuracy.

## 🚀 Usage Instructions

### Prerequisites
```bash
pip install torch torchvision tqdm
```

### Running the Models

#### Model 1 (Basic CNN)
```bash
python main.py --model model1 --epochs 15
```

#### Model 2 (With BatchNorm & GAP)
```bash
python main.py --model model2 --epochs 15
```

#### Model 3 (Advanced with Augmentation)
```bash
python main.py --model model3 --epochs 15
```

### Custom Training
```bash
python main.py --model model3 --epochs 20 --lr 0.01 --batch_size 64 --data_dir ./data
```

## 📈 Training Progress Comparison

### Model 1 Training Curve
- **Epoch 1**: 69.17% → 96.19% (rapid initial learning)
- **Epoch 5**: 97.40% → 98.03% (steady improvement)
- **Epoch 15**: 98.32% → 98.89% (convergence)

### Model 2 Training Curve
- **Epoch 1**: 88.08% → 97.89% (BatchNorm advantage)
- **Epoch 5**: 97.42% → 98.71% (stable progression)
- **Epoch 15**: 98.04% → 99.18% (superior final performance)

### Model 3 Training Curve
- **Epoch 1**: 91.63% → 98.49% (excellent initial learning with augmentation)
- **Epoch 5**: 97.83% → 99.30% (rapid convergence with optimized LR)
- **Epoch 15**: 98.34% → 99.46% (superior final performance)

## 🎓 Learning Outcomes

This project demonstrates several key deep learning concepts:

1. **Architecture Design**: How to build efficient CNNs with limited parameters
2. **Normalization Techniques**: The impact of BatchNorm on training stability
3. **Pooling Strategies**: Global Average Pooling vs. Fully Connected layers
4. **Data Augmentation**: How to improve generalization with transformations
5. **Learning Rate Scheduling**: The importance of adaptive learning rates
6. **Progressive Enhancement**: Building upon previous models' strengths

## 🔮 Future Enhancements

- **Model 4**: Residual connections and attention mechanisms
- **Model 5**: Ensemble methods and model averaging
- **Model 6**: Transfer learning from pre-trained models
- **Model 7**: Neural architecture search for optimal design

## 📝 Conclusion

This project showcases the evolution of CNN design from basic architectures to sophisticated, well-optimized models. Each model builds upon the previous one's learnings, demonstrating how modern deep learning techniques can achieve high accuracy with efficient parameter usage. The progression from 98.89% to 99.46% accuracy illustrates the cumulative impact of architectural improvements, normalization techniques, and advanced training strategies.

The key takeaway is that **thoughtful design and modern techniques often outperform brute-force parameter scaling**, making this an excellent foundation for understanding deep learning best practices.

## 📜 Logged Training Outputs (Excerpts)

Below are brief snippets from the saved logs to illustrate parameter counts and training progress for each model.

### model1_output.log
```text
Model: model1
Total parameters: 7,616
...
Epoch 1
Train Loss: 0.8583 | Train Acc: 69.17% | Test Loss: 0.1172 | Test Acc: 96.19%
...
Epoch 15
Train Loss: 0.0521 | Train Acc: 98.32% | Test Loss: 0.0334 | Test Acc: 98.89%
```

### model2_output.log
```text
Model: model2
Total parameters: 7,798
...
Epoch 1
Train Loss: 0.4042 | Train Acc: 88.08% | Test Loss: 0.0689 | Test Acc: 97.89%
...
Epoch 15
Train Loss: 0.0627 | Train Acc: 98.04% | Test Loss: 0.0277 | Test Acc: 99.18%
```

### model3_output.log
```text
Random seed set to: 42
Total params: 7,762
...
Epoch 1
Train Loss: 0.2859 | Train Acc: 91.63% | Test Loss: 0.0459 | Test Acc: 98.49%
...
Epoch 15
Train Loss: 0.0526 | Train Acc: 98.34% | Test Loss: 0.0181 | Test Acc: 99.46%
```
