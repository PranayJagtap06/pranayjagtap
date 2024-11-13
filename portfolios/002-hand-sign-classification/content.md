## Hand Sign Classification using Convolutional Neural Networks (CNNs) Architecture

<em><a href="https://github.com/PranayJagtap06/ML_Projects/tree/main/Hand_Signs_Classification" target="_blank" rel="noopener noreferrer">Check out my Hand Sign Classification project on GitHub!</a></em>

<em><a href="https://python.plainenglish.io/decoding-hand-signs-a-step-by-step-guide-to-building-a-0-to-5-classifier-using-cnn-and-tensorflow-8b87db221e5e" target="_blank" rel="noopener noreferrer">Read Related Medium Article</a></em>

This project entailed the development of a deep learning model capable of classifying hand signs from 0 to 5 using Convolutional Neural Networks (CNNs). The primary objective was to design a highly accurate and efficient system that can reliably distinguish between different hand signs.

<div align="center">
    <figure>
        <a href="https://pranayjml.odoo.com/web/image/260-729d94da/prediction_5.jpg?access_token=d64993bb-f51b-452b-8113-f236a2b04aef">
            <img class="mk-img" src="https://pranayjml.odoo.com/web/image/260-729d94da/prediction_5.jpg?access_token=d64993bb-f51b-452b-8113-f236a2b04aef" alt="Hand Sign Prediction 5">
        </a>
        <figcaption>Hand Sign Prediction 5</figcaption>
    </figure>
</div>

### Data Preprocessing

To ensure optimal model performance, a data preprocessing phase (model was performing well without data augmentation, so data augmentation was not performed and only data scaling was performed) was undertaken to enhance the quality and consistency of the input image data. This crucial step involved applying normalization to prepare the data for the CNN model.

### Model Architecture and Training

A bespoke CNN architecture was designed and implemented using TensorFlow. A thorough hyperparameter tuning process was conducted to determine the optimal learning rate, which was identified as 0.002 using the Adam optimizer. Subsequently, the final model was built and trained using the optimized hyperparameters.

### Model Evaluation and Performance

The trained model demonstrated exceptional performance on both the validation and test sets. On the validation set, the model achieved an accuracy of 85% and a categorical accuracy of 100%. Notably, the model performed even more impressively on the test set, with an accuracy of 88.33% and a categorical accuracy of 100%. These results unequivocally demonstrate the model's ability to accurately classify hand signs from 0 to 5.

### Conclusion

This project showcases my proficiency in designing and implementing a deep learning model that can accurately classify hand signs using CNNs and TensorFlow. The results underscore the efficacy of the proposed approach and highlight the vast potential of machine learning in image classification tasks.