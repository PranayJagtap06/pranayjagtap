## American Sign Language (ASL) Detection

<em><a href="https://github.com/PranayJagtap06/UFM_ASL_Detection" target="_blank" rel="noopener noreferrer">Check out my ASL Detection project on GitHub!</a></em>

<em><a href="https://american-sign-language-detector.streamlit.app/" target="_blank" rel="noopener noreferrer">Visit Project's Streamlit App</a></em>

ASL Detection is simple computer vision problem, trying to identify the ASL hand signs from the image input using `torchvision`'s `EfficientNet_B0` model. The main objective is to detect the ASL hand signs from the input image. The motivation behind this project is to help people with hearing impairments to communicate more effectively. Practicing `Transfer Learning` with `PyTorch` is also a key aspect of this project. The dataset used is [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) dataset from [Kaggle](https://kaggle.com).

### Model Training Work Flow

A base model with default weights (`torchvision.models.EfficientNet_B0_Weights.DEFAULT`) is used for training. All layers of the model were frozen and the output layer of the model is modified to suite our problem of `out_features=29` (original model comes with `out_features=1000`). The model is trained on a dataset of `29 ASL hand signs`, with a total of `87,000 images`. The model is trained for `10 epochs` with a batch size of `64`. `Adam` optimizer with a learning rate of `0.001` is chosen. And a loss function `CrossEntropyLoss`, with a patience of `5` for early stopping is utilized. Every image before feeding into the model is transformed using `torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()`. 

> ```python
> # weights.transforms()
> ImageClassification(
>     crop_size=[224]
>     resize_size=[256]
>     mean=[0.485, 0.456, 0.406]
>     std=[0.229, 0.224, 0.225]
>     interpolation=InterpolationMode.BICUBIC
> )
> ```

The model is trained on a `Tesla T4` GPU.

> <span style="color: red;">**Note:** The model is trained on a Kaggle machine with a decent GPU, so the training time might vary based on the hardware.</span>

### Model Evaluation

##### <em> Re-Trained `EfficientNet_B0 Model` Summary </em>

> ```pl
> ================================================================================================================================================================
> Layer (type (var_name))                                      Input Shape               Output Shape              Param #                   Trainable
> ================================================================================================================================================================
> EfficientNet (EfficientNet)                                  [64, 3, 224, 224]         [64, 29]                  --                        Partial
> ├─Sequential (features)                                      [64, 3, 224, 224]         [64, 1280, 7, 7]          --                        False
> │    └─Conv2dNormActivation (0)                              [64, 3, 224, 224]         [64, 32, 112, 112]        --                        False
> │    │    └─Conv2d (0)                                       [64, 3, 224, 224]         [64, 32, 112, 112]        (864)                     False
> │    │    └─BatchNorm2d (1)                                  [64, 32, 112, 112]        [64, 32, 112, 112]        (64)                      False
> │    │    └─SiLU (2)                                         [64, 32, 112, 112]        [64, 32, 112, 112]        --                        --
> │    └─Sequential (1)                                        [64, 32, 112, 112]        [64, 16, 112, 112]        --                        False
> │    │    └─MBConv (0)                                       [64, 32, 112, 112]        [64, 16, 112, 112]        (1,448)                   False
> │    └─Sequential (2)                                        [64, 16, 112, 112]        [64, 24, 56, 56]          --                        False
> │    │    └─MBConv (0)                                       [64, 16, 112, 112]        [64, 24, 56, 56]          (6,004)                   False
> │    │    └─MBConv (1)                                       [64, 24, 56, 56]          [64, 24, 56, 56]          (10,710)                  False
> │    └─Sequential (3)                                        [64, 24, 56, 56]          [64, 40, 28, 28]          --                        False
> │    │    └─MBConv (0)                                       [64, 24, 56, 56]          [64, 40, 28, 28]          (15,350)                  False
> │    │    └─MBConv (1)                                       [64, 40, 28, 28]          [64, 40, 28, 28]          (31,290)                  False
> │    └─Sequential (4)                                        [64, 40, 28, 28]          [64, 80, 14, 14]          --                        False
> │    │    └─MBConv (0)                                       [64, 40, 28, 28]          [64, 80, 14, 14]          (37,130)                  False
> │    │    └─MBConv (1)                                       [64, 80, 14, 14]          [64, 80, 14, 14]          (102,900)                 False
> │    │    └─MBConv (2)                                       [64, 80, 14, 14]          [64, 80, 14, 14]          (102,900)                 False
> │    └─Sequential (5)                                        [64, 80, 14, 14]          [64, 112, 14, 14]         --                        False
> │    │    └─MBConv (0)                                       [64, 80, 14, 14]          [64, 112, 14, 14]         (126,004)                 False
> │    │    └─MBConv (1)                                       [64, 112, 14, 14]         [64, 112, 14, 14]         (208,572)                 False
> │    │    └─MBConv (2)                                       [64, 112, 14, 14]         [64, 112, 14, 14]         (208,572)                 False
> │    └─Sequential (6)                                        [64, 112, 14, 14]         [64, 192, 7, 7]           --                        False
> │    │    └─MBConv (0)                                       [64, 112, 14, 14]         [64, 192, 7, 7]           (262,492)                 False
> │    │    └─MBConv (1)                                       [64, 192, 7, 7]           [64, 192, 7, 7]           (587,952)                 False
> │    │    └─MBConv (2)                                       [64, 192, 7, 7]           [64, 192, 7, 7]           (587,952)                 False
> │    │    └─MBConv (3)                                       [64, 192, 7, 7]           [64, 192, 7, 7]           (587,952)                 False
> │    └─Sequential (7)                                        [64, 192, 7, 7]           [64, 320, 7, 7]           --                        False
> │    │    └─MBConv (0)                                       [64, 192, 7, 7]           [64, 320, 7, 7]           (717,232)                 False
> │    └─Conv2dNormActivation (8)                              [64, 320, 7, 7]           [64, 1280, 7, 7]          --                        False
> │    │    └─Conv2d (0)                                       [64, 320, 7, 7]           [64, 1280, 7, 7]          (409,600)                 False
> │    │    └─BatchNorm2d (1)                                  [64, 1280, 7, 7]          [64, 1280, 7, 7]          (2,560)                   False
> │    │    └─SiLU (2)                                         [64, 1280, 7, 7]          [64, 1280, 7, 7]          --                        --
> ├─AdaptiveAvgPool2d (avgpool)                                [64, 1280, 7, 7]          [64, 1280, 1, 1]          --                        --
> ├─Sequential (classifier)                                    [64, 1280]                [64, 29]                  --                        True
> │    └─Dropout (0)                                           [64, 1280]                [64, 1280]                --                        --
> │    └─Linear (1)                                            [64, 1280]                [64, 29]                  37,149                    True
> ================================================================================================================================================================
> Total params: 4,044,697
> Trainable params: 37,149
> Non-trainable params: 4,007,548
> Total mult-adds (Units.GIGABYTES): 24.62
> ================================================================================================================================================================
> Input size (MB): 38.54
> Forward/backward pass size (MB): 6904.20
>  Params size (MB): 16.18
>  Estimated Total Size (MB): 6958.91
>  ================================================================================================================================================================
> ```

<div align="center">
    <figure>
        <a href="/portfolios/005-asl-detector-pytorch/loss_curve.html">
            <iframe 
                src="/portfolios/005-asl-detector-pytorch/loss_curve.html" 
                width="100%" 
                height="775px" 
                frameborder="0">
            </iframe>
        </a>
        <figcaption> <em> Loss Curve </em> </figcaption>
    </figure>
</div>

<div align="center">
    <figure>
        <a href="/portfolios/005-asl-detector-pytorch/accuracy_curve.html">
            <iframe 
                src="/portfolios/005-asl-detector-pytorch/accuracy_curve.html" 
                width="100%" 
                height="775px" 
                frameborder="0">
            </iframe>
        </a>
        <figcaption> <em> Accuracy Curve </em> </figcaption>
    </figure>
</div>

##### <em> Trained Model Classification Report </em>

> ```pl
> Classification Report:
>               precision    recall  f1-score   support
> 
>            A       0.97      0.97      0.97      3000
>            B       0.98      0.98      0.98      3000
>            C       0.99      0.99      0.99      3000
>            D       0.99      0.99      0.99      3000
>            E       0.97      0.97      0.97      3000
>            F       0.99      0.99      0.99      3000
>            G       0.98      0.97      0.98      3000
>            H       0.98      0.98      0.98      3000
>            I       0.96      0.96      0.96      3000
>            J       0.98      0.97      0.98      3000
>            K       0.97      0.97      0.97      3000
>            L       0.99      0.98      0.99      3000
>            M       0.95      0.95      0.95      3000
>            N       0.96      0.96      0.96      3000
>            O       0.99      0.99      0.99      3000
>            P       0.98      0.98      0.98      3000
>            Q       0.98      0.98      0.98      3000
>            R       0.93      0.93      0.93      3000
>            S       0.92      0.93      0.93      3000
>            T       0.94      0.94      0.94      3000
>            U       0.91      0.91      0.91      3000
>            V       0.92      0.92      0.92      3000
>            W       0.94      0.95      0.94      3000
>            X       0.92      0.92      0.92      3000
>            Y       0.96      0.95      0.96      3000
>            Z       0.96      0.96      0.96      3000
>          del       0.98      0.99      0.99      3000
>      nothing       1.00      1.00      1.00      3000
>        space       0.98      0.99      0.99      3000
> 
>     accuracy                           0.96     87000
>    macro avg       0.96      0.96      0.96     87000
> weighted avg       0.96      0.96      0.96     87000
> ```

<div align="center">
    <figure >
        <a href="/portfolios/005-asl-detector-pytorch/confusion_matrix.html">
            <iframe 
                src="/portfolios/005-asl-detector-pytorch/confusion_matrix.html" 
                width="100%" 
                height="950px" 
                frameborder="0">
            </iframe>
        </a>
        <figcaption> <em> Confusion Matrix </em> </figcaption>
    </figure>
</div>

### Conclusion

The American Sign Language (ASL) Detection project successfully demonstrates the application of computer vision techniques to recognize hand signs using the `EfficientNet_B0` model. By leveraging transfer learning with `PyTorch`, the model was fine-tuned on a substantial dataset of 87,000 images representing `29` ASL hand signs. 

The architecture of the model, which includes a modified output layer tailored for our specific classification task, has proven effective. The training process utilized a batch size of 64 and was conducted over `10` epochs, employing the Adam optimizer and CrossEntropyLoss for optimal performance. 

The evaluation metrics from the classification report indicate a high level of accuracy, with an overall <span style="color: #04cdfa"><strong>accuracy of 96%</strong></span>. Each class demonstrated <span style="color: #04cdfa"><strong>strong precision and recall</strong></span>, particularly for the classes representing the letters A through Z, as well as the special signs for "del," "nothing," and "space." The confusion matrix further illustrates the model's capability to distinguish between different signs, with minimal misclassifications.

In conclusion, this project not only showcases the potential of deep learning in enhancing communication for individuals with hearing impairments but also serves as a practical example of implementing advanced machine learning techniques in real-world applications. Future work may involve expanding the dataset, improving model robustness, and exploring real-time detection capabilities.



