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

# Metrics for Classifiers (Multicategorical Outcomes)

Beyond models predicting binary outcomes, a confusion matrix can be used to evaluate models predicting multicategorical outcomes. The confusion matrix for multicategorical outcomes could look like this:

| Actual \ Predicted | Lung Cancer | Breast Cancer | Skin Cancer |
|--------------------|-------------|---------------|-------------|
| Lung Cancer        | 50          | 8             | 5           |
| Breast Cancer      | 7           | 75            | 3           |
| Skin Cancer        | 4           | 2             | 80          |

Similar to the matrix for binary outcomes, the matrix for multicategorical outcomes can be used to calculate accuracy, sensitivity, specificity, and precision. However, certain points need to be adjusted using two primary methods:

## Macro-Averaging
**This method calculates each metric separately for each class and then takes the average**. This approach gives equal importance to each class, regardless of its size. If your data has class imbalance, macro-averaging is reliable in evaluating model performance because the metric is calculated independently for each class before taking the average, giving equal weight to each class regardless of its size. However, just like metrics for binary outcomes, certain metrics such as accuracy could be biased by class imbalance. For macro-averaging, each class is given equal weight, so the performance of minority classes can disproportionately affect the overall accuracy. Note that in cases where classes are balanced, both micro- and macro-averaging will yield similar results for most metrics.

## Micro-Averaging
This method considers the contributions of all classes to compute the average metric. If you have class imbalance, this approach may not be appropriate and could be dominated by the majority class. Although micro-averaging might not be a good method to evaluate the performance of each class separately if you have a class imbalance, the method might be appropriate for evaluating the overall effectiveness of the model, especially if the majority class is the most important. This is typically the case in banking or consumer application industries. 

In summary,if you care more about all classes being detected, macro-averaging is the best option. If you care more about overall model performance, micro-averaging is the better option. 

## Let's take a look at an example using the confusion matrix above:
**Accuracy Calculation for Each Class:**
* Lung Cancer: (50 / (50 + 8 + 5)) ≈ 79%
* Breast Cancer: (75 / (7 + 75 + 3)) ≈ 88%
* Skin Cancer: (80 / (4 + 2 + 80)) ≈ 93%

**Macro-Averaging:**
Overall Accuracy: (79% + 88% + 93%) / 3 ≈ 86.67%

**Micro-Averaging:**
* Total Correct Predictions: 50 (i.e., Lung) + 75 (i.e., Breast) + 80 (i..e, Skin) = 205
* Total Predictions Made: 63 (Lung) + 85 (Breast) + 86 (Skin) = 234
* Micro-Averaged Accuracy: 205 / 234 ≈ 87.61%


Now you have an understanding of how to calculate accuracy using macro- and micro-averaging. This concept can be applied to other metrics such as precision, recall, and specificity.

# Evaluation Metrics for Continuous Outcomes

For continuous outcomes, the model aims to predict numeric values instead of categorical ones. The goal of all regressors (i.e., models predicting continuous outcomes) is to minimize the distance between observed and predicted values, as shown in the equation: 

*e*<sub>*i*</sub> = *y*<sub>*i*</sub> - $\hat{y}$<sub>*i*</sub>
​	

To better understand errors and how they are produced, consider one of the simplest ML algorithms: linear regression. This can be represented by the equation:

*y* = *β*<sub>*0*</sub> + *β*<sub>*1*</sub>*X*<sub>*1*</sub> + *β*<sub>*2*</sub>*X*<sub>*2*</sub> + *ϵ*

Initially, the algorithm does not know the best values for *β*<sub>*1*</sub> and *β*<sub>*2*. It begins by plugging in random numbers for these parameters and iteratively adjusts them to minimize *ϵ*  (i.e.,the distance between observed and predicted values). When the process is performed iteractively, you can get something like the plot below: 

<img width="591" alt="Screen Shot 2024-06-05 at 12 45 50 PM" src="https://github.com/KayChansiri/Demo_Performance_Metrics/assets/157029107/2895c6ea-f45b-4371-af7d-055fa9af5a7a">


From the plot, you can see the best pair of values for *β*<sub>*1*</sub> and *β*<sub>*2* that minimizes *ϵ* would be at somewhere at the bottom right corner where I circled. When you substitute these values back into the equation *β*<sub>*0*</sub> + *β*<sub>*1*</sub>*X*<sub>*1*</sub> + *β*<sub>*2*</sub>*X*<sub>*2*</sub> + *ϵ*, and replace *X*<sub>*1*</sub> and *X*<sub>*2*</sub> with observed values across all instances, you will get the error for each instance. These errors can be used to evaluate the model using the following methods:

## Mean Absolute Error (MAE)

<img width="277" alt="Screen Shot 2024-06-05 at 12 58 07 PM" src="https://github.com/KayChansiri/Demo_Performance_Metrics/assets/157029107/31d361b6-29d0-4141-9c43-32a287916e41">

 
* This is calculated by summing the absolute values of errors across all samples.
* Some people prefer using Mean Squared Error (MSE) over MAE because MSE penalizes larger errors more severely and avoids issues with calculus that may arise with MAE.

## Mean Squared Error (MSE)

<img width="306" alt="Screen Shot 2024-06-05 at 12 58 56 PM" src="https://github.com/KayChansiri/Demo_Performance_Metrics/assets/157029107/90ffb083-7ecf-4186-b972-3f1cfda921c5">


## Root Mean Squared Error (RMSE)**
  
<img width="357" alt="Screen Shot 2024-06-05 at 12 59 00 PM" src="https://github.com/KayChansiri/Demo_Performance_Metrics/assets/157029107/d68d0bf8-4172-455e-ad42-af6e23269b3f">

# K-Fold Cross-Validation

Now that you learned all of important evaluatiom metrics for machine learning, it is noteworthy that it would never be sufficient to evaluate a model just once and then assume it performs well or poorly. 
> A standard practice in machine learning is to use multiple samples, ideally independent from each other, and average the evaluation metrics to assess overall model performance. 

* The process is known as **K-fold cross-validation**, where data is separated into *n*  sets. *n-1* sets are used as the training set and the remaining set as the testing set. Some may suggest having an additional validation set to fine-tune the model before final testing and deployment. You can read more about cross-validation techniques in [my previous post](https://github.com/KayChansiri/demo_random_forest-), which mainly discusses ensemble techniques but also covers cross-validation in detail.

* In addition to obtaining average metrics for regressors (i.e., models predicting continuous outcomes) or majority votes for classifiers (i.e., models predicting categorical outcomes), we may also use statistical tests to determine if one model performs better than another. A common method is the paired t-test to see if the averages across models are statistically different. This is suitable because K-fold cross-validation samples are dependent on each other, being resampled from the same population.

* However, some researchers argue that the paired sample t-test may not be ideal due to its sensitivity to outliers. Alternatives such as Wilcoxon’s signed-rank test or DeLong's test are recommended depending on the metric being compared. I encourage you to read the article by Rainio, Teuho, and Keen, which explains which statistical methods to use for comparing evaluation metrics across models. Their Figure 3 is particularly informative. You can find the article here: [Nature Article](https://www.nature.com/articles/s41598-024-56706-x).
*In summary, I would say that the choice of statistical test depends on your data (e.g., evaluation metric values for parametric or non-parametric metrics), the number of models you want to compare (two or more), and whether you are comparing means or variances of the evaluation outcomes.


There you have it! I hope this post helps you learn the basics of evaluation metrics for machine learning. Let me know in the comments below if you have any questions!







