## Mobile Phone Price Classification

<em><a href="https://github.com/PranayJagtap06/UFM_Mobile_Phone_Pricing" target="_blank" rel="noopener noreferrer">Check out my Mobile Phone Price Classification project on GitHub!</a></em>

<em><a href="https://mob-price-range-classifier.streamlit.app/" target="_blank" rel="noopener noreferrer">Visit Project's Streamlit App</a></em>

This project is a simple mobile phone price classification model using `Scikit-Learn` models. The prime objective of this project is to classify mobile phones into different price ranges based on their features. The motivation behind this project is to help users make informed decisions while purchasing a mobile phone. And also to learn implementation of Scikit-Learn's `Pipeline` and `GridSearchCV` classes.

### Exploratory Data Analysis

A detail exploratory data analysis is performed on the dataset to understand the distribution and relationship between the features and the target variable.It was found that the dataset is balanced, and features like `RAM`, `internal memory`, and `4G availability` have a significant impact on the price of mobile. While `battery power`, `screen size` and other features have a moderate to little impact.

<div align="center">
    <figure>
        <a href="/portfolios/004-mobile-price-classification/mobile_phone_scatter_plot.html">
            <iframe 
                src="/portfolios/004-mobile-price-classification/mobile_phone_scatter_plot.html" 
                width="100%" 
                height="470px" 
                frameborder="0">
            </iframe>
        </a>
        <figcaption> <em> Mobile Phone Scatter Plot </em> </figcaption>
    </figure>
</div>

### Model Training Work Flow

Trained 3 different models, `LogisticRegression`, `KNeighborsClassifier`, & `RandomForestClassifier`, using Scikit-Learn's `Pipeline` and `GridSearchCV` classes. All the models were trained on the same dataset and were evaluated using the same metrics. The best model was selected based on the highest accuracy & F1-score. The model with the highest accuracy (91%) & F1-score (avg: 0.91) was `RandomForestClassifier`. The all models' performances were evaluated using the `confusion matrix`, `classification report`, `precision recall curve`, and `roc curve`. The results are shown in next section. Finally, all the models and plots were logged using `DAGsHub` and `mlflow` library for future use and deployment.

### Model Evaluation

##### <em> 1. Logistic Regression Model & Metrics </em>

```python
# Logistic Regression Model
GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('standardscaler', StandardScaler()),
                                       ('logisticregression',
                                        LogisticRegression(max_iter=1000,
                                                           solver='liblinear'))]),
             param_grid={'logisticregression__C': [0.01, 0.1, 1, 10, 100],
                         'logisticregression__penalty': ['l1', 'l2']},
             scoring='accuracy')

# Best Estimator
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('logisticregression',
                 LogisticRegression(C=10, max_iter=1000, penalty='l1',
                                    solver='liblinear'))])
```

```pl
Training Accuracy: 0.8938
Testing Accuracy: 0.84
```

```pl
0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost

Test Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      1.00       100
           1       0.72      0.68      0.70       100
           2       0.68      0.70      0.69       100
           3       0.96      0.98      0.97       100

    accuracy                           0.84       400
   macro avg       0.84      0.84      0.84       400
weighted avg       0.84      0.84      0.84       400
```

<div align="center">
    <figure>
        <a href="/portfolios/004-mobile-price-classification/confusion_matrix_log_reg.html">
            <iframe 
                src="/portfolios/004-mobile-price-classification/confusion_matrix_log_reg.html" 
                width="100%" 
                height="470px" 
                frameborder="0">
            </iframe>
        </a>
        <figcaption> <em> Confusion Matrix: Logistic Regression Model </em> </figcaption>
    </figure>
</div>

##### <em> 2. KNeighborsClassifier Model & Metrics </em>

```python
# KNN Model
GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('standardscaler', StandardScaler()),
                                       ('kneighborsclassifier',
                                        KNeighborsClassifier())]),
             param_grid={'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11,
                                                               13, 15],
                         'kneighborsclassifier__weights': ['uniform',
                                                           'distance']},
             scoring='accuracy')

# Best Estimator
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('kneighborsclassifier',
                 KNeighborsClassifier(n_neighbors=13, weights='distance'))])
```

```pl
Training Accuracy: 1
Testing Accuracy: 0.5750
```

```pl
0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost
Test Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.68      0.73       100
           1       0.41      0.45      0.43       100
           2       0.44      0.47      0.45       100
           3       0.72      0.70      0.71       100

    accuracy                           0.57       400
   macro avg       0.59      0.57      0.58       400
weighted avg       0.59      0.57      0.58       400
```

<div align="center">
    <figure>
        <a href="/portfolios/004-mobile-price-classification/confusion_matrix_knn.html">
            <iframe 
                src="/portfolios/004-mobile-price-classification/confusion_matrix_knn.html" 
                width="100%" 
                height="470px" 
                frameborder="0">
            </iframe>
        </a>
        <figcaption> <em> Confusion Matrix: KNeighborsClassifier Model </em> </figcaption>
    </figure>
</div>

##### <em> 3. Random Forest Classifier Model </em>

```python
# Random Forest Classifier Model
GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('standardscaler', StandardScaler()),
                                       ('randomforestclassifier',
                                        RandomForestClassifier(n_jobs=-1))]),
             param_grid={'randomforestclassifier__max_depth': [None, 5, 10, 20],
                         'randomforestclassifier__max_features': [10, 19,
                                                                  'sqrt',
                                                                  'log2'],
                         'randomforestclassifier__min_samples_leaf': [1, 2, 4],
                         'randomforestclassifier__min_samples_split': [2, 5,
                                                                       10],
                         'randomforestclassifier__n_estimators': [100, 150, 250,
                                                                  300]},
             scoring='accuracy')

# Best Estimator
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=20, max_features=19,
                                        min_samples_leaf=4, min_samples_split=5,
                                        n_estimators=150, n_jobs=-1))])
```

```pl
Training Accuracy: 0.9781
Testing Accuracy: 0.91
```

```pl
0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost
Test Classifier Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.95      0.95       100
           1       0.86      0.87      0.87       100
           2       0.86      0.87      0.87       100
           3       0.96      0.95      0.95       100

    accuracy                           0.91       400
   macro avg       0.91      0.91      0.91       400
weighted avg       0.91      0.91      0.91       400
```


<div align="center">
    <figure>
        <a href="/portfolios/004-mobile-price-classification/confusion_matrix_rf.html">
            <iframe 
                src="/portfolios/004-mobile-price-classification/confusion_matrix_rf.html" 
                width="100%" 
                height="470px" 
                frameborder="0">
            </iframe>
        </a>
        <figcaption> <em> Confusion Matrix: Random Forest Classifier Model </em> </figcaption>
    </figure>
</div>

### Conclusion

Based on the evaluation of the models used in the mobile phone price classification project, we can draw several conclusions regarding their performance and effectiveness.

The <span style="color: #04cdfa"><strong>Random Forest Classifier</strong></span> emerged as the best-performing model, achieving an impressive <span style="color: #04cdfa"><strong>testing accuracy of 91%</strong></span> and an average <span style="color: #04cdfa"><strong>F1-score of 0.91</strong></span>. This model demonstrated <span style="color: #04cdfa"><strong>strong precision and recall</strong></span> across all price categories, particularly excelling in the low and very high-cost classifications. The confusion matrix for the Random Forest model indicates that it effectively minimizes misclassifications, making it a reliable choice for this classification task.

In contrast, the <span style="color: #04cdfa"><strong>Logistic Regression Model</strong></span> achieved a <span style="color: #04cdfa"><strong>testing accuracy of 84%</strong></span> with an <span style="color: #04cdfa"><strong>F1-score of 0.84</strong></span>. While it performed well in identifying low-cost phones, its performance in the medium and high-cost categories was less robust, indicating potential areas for improvement. The confusion matrix highlights some misclassifications, particularly in the medium-cost category, which could be addressed through further feature engineering or model tuning.

The <span style="color: #04cdfa"><strong>KNeighborsClassifier</strong></span> model, however, showed the least effectiveness, with a <span style="color: #04cdfa"><strong>testing accuracy of only 57%</strong></span>. Despite achieving perfect training accuracy, it struggled to generalize to the test set, resulting in lower precision and recall across all categories. This suggests that the KNN model may not be suitable for this particular dataset without significant adjustments to its parameters or the inclusion of additional features.

Overall, the Random Forest Classifier stands out as the most effective model for mobile phone price classification, providing a solid foundation for future enhancements and potential deployment. The insights gained from this project not only contribute to better decision-making for consumers but also serve as a valuable learning experience in applying machine learning techniques using `Scikit-Learn`. The model was later deployed using <span style="color: #04cdfa"><strong>Streamlit</strong></span>.

