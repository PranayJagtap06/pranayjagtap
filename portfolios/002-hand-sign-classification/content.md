## Hand Sign Classification using Convolutional Neural Networks (CNNs) Architecture

<em><a href="https://github.com/PranayJagtap06/ML_Projects/tree/main/Hand_Signs_Classification" target="_blank" rel="noopener noreferrer">Check out my Hand Sign Classification project on GitHub!</a></em>

<em><a href="https://python.plainenglish.io/decoding-hand-signs-a-step-by-step-guide-to-building-a-0-to-5-classifier-using-cnn-and-tensorflow-8b87db221e5e" target="_blank" rel="noopener noreferrer">Read Related Medium Article</a></em>

This project entailed the development of a deep learning model capable of `classifying hand signs from 0 to 5 using Convolutional Neural Networks (CNNs)`. The primary objective was to design a highly accurate and efficient system that can reliably distinguish between different hand signs.

<div align="center">
    <figure>
        <a href="portfolios/002-hand-sign-classification/5-hand-sign-prediction.png">
            <img class="mk-img" src="portfolios/002-hand-sign-classification/5-hand-sign-prediction.png" alt="Hand Sign Prediction 5">
        </a>
        <figcaption> <em> Hand Sign Prediction 5 </em> </figcaption>
    </figure>
</div>

### Data Preprocessing

To ensure optimal model performance, a data preprocessing phase (model was performing well without data augmentation, so data augmentation was not performed and only data scaling was performed) was undertaken to enhance the quality and consistency of the input image data. This crucial step involved applying normalization to prepare the data for the CNN model.

### Model Architecture and Training

A bespoke CNN architecture was designed and implemented using `TensorFlow`. A thorough hyperparameter tuning process was conducted to determine the optimal learning rate, which was identified as `0.002` using the Adam optimizer. Subsequently, the final model was built and trained using the optimized hyperparameters.

### Model Evaluation and Performance

The trained model demonstrated exceptional performance on both the validation and test sets. On the validation set, the model achieved an `accuracy of 85%` and a `categorical accuracy of 100%`. Notably, the model performed even more impressively on the test set, with an `accuracy of 88.33%` and a `categorical accuracy of 100%`. These results undoubtedly demonstrate the model's ability to accurately classify hand signs from 0 to 5.

### Conclusion

The results underscore the efficacy of the proposed approach and highlight the vast potential of machine learning in image classification tasks. This project showcases my proficiency in designing and implementing a deep learning model that can accurately classify hand signs using CNNs and TensorFlow. The model's impressive performance on both the validation and test sets underscores its reliability and robustness.