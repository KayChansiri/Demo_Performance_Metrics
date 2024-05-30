# Performance Evaluation Metrics

Hello, all data scientists and researchers! Today, I will talk about one of the most important topics in machine learning - performance evaluation metrics. We will never know if a model is effective unless we deeply understand which metric to choose to evaluate the model. I will discuss metrics in two categories: 1) metrics for regressors, and 2) metrics for classifiers.

# Metrics for Classifiers
## Introduction 
This type of metric is used for models predicting categorical outcomes, which could be binary (e.g.,predicting if patients have or do not have cancer) or multi-categorical (e.g., predicting if customers would buy Porsche, Lexus, or Mercedez as their first car). Typically, classifiers categorize data into specific groups/labels. However, certain algorithms, such as logistic regression, produce a continuous output that represents the probability of a case belonging to a class, typically ranging from 0 to 1. This probability is often converted into a categorical outcome for easier interpretation and decision-making by applying a threshold. For instance, if the probability of a patient having cancer is 0.78, and our cutoff is 0.5, we would classify this case as '1' (having cancer). Now that you have an idea of what a classifier is, let's discuss how many metrics are available to evaluate this type of model.

### 1. Confusion Matrix

* A Confusion Matrix is a standard evaluation metric used for classifiers, including those targeting binary and multicategorical outcomes. The matrix is typically presented as a table indicating true positives, true negatives, false positives, and false negatives.
* For example, imagine you build a model for cancer detection example, your confusion matrix may look like this:

|                        | Predicted: No Cancer | Predicted: Cancer |
|------------------------|----------------------|-------------------|
| **Actual: No Cancer**  | 800 (True Negative)   | 200 (False Positive)|
| **Actual: Cancer**     | 27 (False Negative)  | 123 (True Positive)|

* The table indicates that there are 10,000 observations that do not have cancer. The classifier accurately predicts '0' for 8,000 of them (i.e., true negatives: TN) and inaccurately predicts '1' for 2,000 of them (false positives: FP). Similarly, there are 150 observations that have cancer. The classifier accurately predicts '1' for 123 of them (true positives: TP) and inaccurately predicts '0' for 27 of them (false negatives: FP).

* However, the matrix alone may not fully illustrate a model’s performance. You might then wonder: What can we do with this confusion matrix? The answer is that you can use the values in the table to calculate key metrics such as accuracy (i.e., the percentage of correctly classified instances), sensitivity or recall (i.e., the proportion of truly positive instances that are correctly identified), specificity (i.e., the proportion of truly negative instances that are correctly identified), and precision (i.e., the proportion of instances classified as positive that are truly positive).

### 2. Accuracy
* As mentioned earlier, accuracy assesses the percentage of correctly classified samples. The metric can be represented by the following formular:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

* This metric is particularly useful when there is a class balance in the outcome, such as having a binary outcome where the ratio of 0 and 1 is closed to 50:50. This is becasue accuracy considers all instances in the denominator.
* However, the metric can be biased if you have an imbalanced outcome. In the cancer example mentioned previosly, 10,000 instances do not have cancer whereas only 150 instances have cancer. 

|                        | Predicted: No Cancer | Predicted: Cancer |
|------------------------|----------------------|-------------------|
| **Actual: No Cancer**  | 8000 (True Negative)   | 2000 (False Positive)|
| **Actual: Cancer**     | 27 (False Negative)  | 123 (True Positive)|

* If you use the formular, the accuracy of the model is 8,000 + 123/ 8,000 + 2,000 + 27 + 123  = ~80%, indicating that the model has a good performance, despite not having learned anything meaningful about the minority class or cases with cancer. 

### 3. Sensitivity (AKA Recall)

* You may wonder if accuracy does not work well with a class imbalance, which metric should you then use. The answer is sensitivity.
* Sensitivity measures how often an algorithm correctly identifies true positives from all the actual positive samples.The metric can be represented by the following formula:

Sensitivity (Recall): TP / (TP + FN)

* The values range from 0 to 1, with higher scores indicating more precise detection of positive cases.
* The metric works well for problems with imbalanced classes, especially when the cost of false negatives is high. Consider the cancer detection example; the sensitivity score here is 123/(27 + 123) ≈ 82, which indicates that the model learns quite well to predict cancer cases out of all actual cancer cases, even though those are the minority class.
* Sensitivity is a desirable metric, **especially when the cost of a false negative is high**. In other words, it should be used as the gold standard metric when you do not want to overlook false negatives or cases where people actually have cancer but the model predicts they do not.
* Note that sensitivity is often reported along with precision or specificity to provide a more comprehensive evaluation.

### 4. Specificity

* This metric assesses the truly negative samples, as the formula below illustrates:

Specificity: TN / (TN + FP)

* Similar to sensitivity, specificity is a good evaluation metric if there is an imbalance between the number of real positive and negative instances.

### 5. Precision
* Precision assesses the proportion of correct positive predictions made by the model when it predicts that the outcome is positive. The fomular is as below:
* Precision = TP/ TP + FP
* The precision scores range from 0 to 1, with higher scores indicating greater accuracy in predicting positive instances.
* Like sensitivity, precision is desirable, **especially when the cost of a false positive is high.** For example, imagine you work in real estate and you would like to offer discounts to previous customers who tend to be serious about buying their next home only, as it would be impractical and costly for your company to offer discounts to everyone. In this case, you may want to consider specificity, which integrates false positives into the formula.
* Despite the advantage of considering false positives, the drawback of precision is that it does not account for false negatives, i.e., customers who are wrongly classified by the model as unlikely to buy but actually would buy. If you use this metric, you may miss targeting these potential buyers.
* Choosing between precision or recall depends on what aspect you value more. Precision is a suitable metric when you care more about *being correct* when assigning the positive class (i.e., when a false positive is costly). Recall is preferable when you care more about *capturing all possible positive cases* (i.e., when missing a positive case has serious consequences), such as in cancer detection or public health interventions where the stakes involve significant financial costs or life-and-death situations. On the other hand, precision could be chosen when the cost of incorrectly assigning positives is high, such as in banking, finance, commercials, or real estate. 

### 6. F-1 scores
* If you care about both precision and recall, you may consider F-1 scores as your evaluation metric.
* The score can be thought of as a harmonic mean of precision and recall, ranging from 0 to 1.  When F1 score is 1, it means that we have perfect precision and recall. When F 1 cores is 0, it means we have bad precision and recall.
* The formular is: F1 = 2 x precision x recall/precision + recall

### 7. Area under the curve (AUC) (also known as the area under the Receiver Operating Characteristic (ROC) curve)
* The AUC is a plot that represents the dynamic relationship between sensitivity (recall) and specificity. When you have a binary outcome for which you need to identify a threshold to categorize the outcome probabilities into 0 or 1, the AUC can help in identifying the best cut-off values.
* As I mentioned earlier, the most common cutoff point is 0.5. However, this also depends on the nature of the project you are working on. For example, if you are working on a project where a lower threshold should be set, such as estimating whether a person should be isolated during a new pandemic, the threshold of 0.3 might be used. You wouldn’t want to use a higher threshold and then let people walk out, potentially spreading the virus!
*  Imagine that you are working with a cancer detection dataset and you would like to see if changing the cutoff point would impact the sensitivity or specificity, starting from using a standard cutoff point which is 0.5. You can plot the AUC plot that would look something like the following:

 <img width="646" alt="Screen Shot 2024-05-30 at 4 46 38 PM" src="https://github.com/KayChansiri/Demo_Performance_Metrics/assets/157029107/406e08b6-c4b4-4f24-9fe6-b57281d58e14">
:

* According to the plot, the threshold may not be visibly specified. However, imagine that the threshold is located somewhere on the curve. If you set a very low threshold (e.g., 0.1), there is a high likelihood that most cases would be classified as positive. If your model performs well at this threshold, your true positive rate would be high (closer to 1) and your false positive rate (i.e., cases classified as positive but are actually negative) would also be high, or closer to 1. This is because more negatives are incorrectly labeled as positives.
* Therefore, this liberal 0.1 threshold would likely be positioned closer to the upper right corner of the ROC curve. Selecting this threshold indicates that you prioritize sensitivity (or recall) more than specificity (also known as the true negative rate). This threshold is suitable in scenarios where missing true positives (e.g., failing to detect a serious disease like cancer) is more critical than dealing with false positives, such as cases where people might be misdiagnosed with cancer. It is better to be safe than sorry.
* Conversely, if you set a more conservative threshold like 0.8, most cases would be classified as negative. Consequently, both your true positive and false positive rates would be lower, as most cases would be classified as negative. Thus, your more conservative 0.8 threshold would appear towards the lower left part of the ROC curve. Selecting this threshold means that you are prioritizing specificity over sensitivity. This approach could be appropriate in scenarios where false positives lead to high costs or consequences, such as avoiding unnecessary medical treatments based on false diagnoses when insurance may not cover the treatment costs.
* You can use `roc_curve` in Python to see the array of thresholds that correspond to each point on the ROC curve, giving you a better idea of which threshold corresponds with which true positive or false positive values.
* Note that the closer the AUC is to 1, the more predictive power the model has. Conversely, the closer the AUC is to 0.5, the closer the predictive power is to random guessing (i.e., the model does not effectively categorize positive and negative cases from each other).




