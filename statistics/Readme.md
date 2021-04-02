## Accuracy, F-1, Precision, and Recall

In this section of the data science toolboox we focus on various evaluation metrics. These statstics are important to understand the how your model is performing based on the output it produces. All these evaluation metrics are part of the confusion matrix, which is an essential way to evaluate success of model. To further understand confusion matrices let's establish an understanding of more fundamentals concepts:

### True Positives (TP):
These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. E.g. if actual class value indicates that this image is a cat and predicted class tells you the same thing.

### True Negatives (TN):
These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. E.g. if actual class says this is not a cat and predicted class tells you the same thing.

### False Positives (FP):
When actual class is no and predicted class is yes. E.g. if actual class says this is a cat but predicted class tells you that this image is not a cat.

### False Negatives (FN):
When actual class is yes but predicted class in no. E.g. if actual class value indicates that this is not a cat and predicted class tells you that it is a cat.

Accuracy, Precisioon, F-1, and recall are all mathemtical combinations and equation of the above four concepts. Let's look at the concepts at these concepts and their equations:

### Accuracy
Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.

Accuracy = TP+TN/(TP+FP+FN+TN)

### Precision
Precision refers to how close measurements are to each other, it describes the variation you see when you measure the same part repeatedly with the same device.  

Precision = TP/TP+FP

### Recall
Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes

Recall = TP/(TP+FN)

![alt text](https://github.com/Headstorm/data-science-toolbox/blob/statistics/AccuracyAndPrecision.png?raw=true)

### F-1
F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy

F1 = 2*(Recall * Precision)/ (Recall + Precision)



## Libraries
Most of the actual computation is done through SKLearn. It is a very extensive (and very useful) Python library to compute most statistical variants/measures needed. It is typically integrated with other useful data science libraries like Numpy and TensorFlow. The visualization is done through Seaborn, another useful Python library that's typically used in tandem with other DS libraries to help understand results better.





#### Documentation, Resources and further reading:
https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/