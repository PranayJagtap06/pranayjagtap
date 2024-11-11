## Hand Sign Classification using Transfer Learning

<em><a href="https://github.com/PranayJagtap06/ML_Projects/tree/main/Hand_Signs_Classification" target="_blank" rel="noopener noreferrer">Check out my Hand Sign Classification project on GitHub!</a></em>

<em><a href="https://medium.com/@pranayjagtap/hand-signs-classification-with-efficientnetb0-transfer-learning-f1506084238b" target="_blank" rel="noopener noreferrer">Read Related Medium Article</a></em>

This project is next-in-line to the Hand Sign Classification with CNN project. This project focuses on creating a strong image classification model that can identify six unique hand signs (0 to 5). The model uses the advantage of transfer learning with the EfficientNetB0 architecture to achieve high accuracy.

### Data Prepsocessing

The image dataset went through detailed preparation to boost model function. Images were adjusted to a set size of (224, 224, 3) to meet the input needs of the EfficientNetB0 model. Normalization was left out since the EfficientNetB0 design includes a built-in normalization layer making the preparation process simpler.

### Model Architecture & Training

The model architecture was carefully crafted using the Keras Functional API, with EfficientNetB0 serving as the base model. The model architecture consists of:

 - An input layer receiving images of size (224, 224, 3).
 - A data augmentation layer for data augmentation.
 - The pre-trained EfficientNetB0 model serving as the feature extractor.
 - A global average pooling layer to reduce the spatial dimensions of the feature maps.
 - A dense output layer with 6 neurons, one for each hand sign class, with softmax activation.


A strategic three-step approach was employed for model building and training:

 - <strong>Base Model Development</strong>: First, a base model was built to include a data augmentation layer and EfficientNetB0. This model was trained using 10% of the training data to implement feature extraction transfer learning.
 - <strong>Fine-Tuning</strong>: The base model went through fine-tuning by making the top 10 layers of EfficientNetB0 trainable and reducing the learning rate by 10%. This process was done on 10% of the training data.
 - <strong>Full Dataset Training</strong>: At last, the optimised model now fine-tuned, was trained using 100% of the training data to improve performance.

### Model Evaluation & Performance

The trained model demonstrates impressive performance on the test set, achieving:

 - Test set loss: 28.21%
 - Test set accuracy: 93.33%
 - Test set Categorical Accuracy: 1.0
 - Test set AUC-ROC score: 96.00%
 - Test set Classification Report:

    | Hand Signs | Precision | Recall | F1-Score |
    | ---- | ---- | ---- | ---- |
    | Hand Sign 0 | 1.00 | 1.00 | 1.00 |
    | Hand Sign 1 | 0.90 | 0.90 | 0.90 |
    | Hand Sign 2 | 0.90 | 0.90 | 0.90 |
    | Hand Sign 3 | 1.00 | 1.00 | 1.00 |
    | Hand Sign 4 | 0.83 | 1.00 | 0.91 |
    | Hand Sign 5 | 1.00 | 0.80 | 0.89 |

The classification report reveals high precision, recall, and F1-score across all hand sign classes, indicating the model's ability to accurately classify each category.

The final model's performance surpasses the previous CNN architecture base model by achieving 5% lower loss and 5% higher accuracy, along with a 3% higher AUC-ROC score. This significant improvement highlights the effectiveness of the transfer learning approach and the chosen model architecture.

### Conclusion

This project showcases my proficiency in building and training a high-performing transfer learning model for image classification. The proposed approach, utilising EfficientNetB0 and a three-step fine-tuning strategy, effectively leverages pre-trained knowledge to achieve state-of-the-art results on the hand sign classification task. The model's impressive performance demonstrates the power of transfer learning in accelerating the development of accurate and efficient image recognition systems.