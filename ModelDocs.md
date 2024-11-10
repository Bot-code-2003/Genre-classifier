Ensemble Model Overview:
python
Copy code
# Define classifiers
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Ensemble Model: Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('lr', lr_model),
    ('rf', rf_model),
    ('gb', gb_model)
], voting='soft')
1. Logistic Regression (lr_model)
Description:

Logistic Regression is a linear model used for binary and multi-class classification tasks. It calculates the probability of a data point belonging to a class using the logistic function.
Key Parameters:

class_weight='balanced': Automatically adjusts the weight of each class based on its frequency. This helps address class imbalance.
max_iter=1000: Increases the maximum number of iterations for the optimization algorithm, ensuring the model converges even for complex datasets.
Strengths:

Efficient and interpretable.
Works well for linearly separable data.
Handles class imbalance well when class_weight='balanced' is used.
Weaknesses:

Performs poorly on non-linear data.
May not capture complex patterns in the text data.
2. Random Forest Classifier (rf_model)
Description:
Random Forest is an ensemble of decision trees where each tree is trained on a random subset of the data (using bootstrapping). The final prediction is made by averaging the predictions of all trees (for regression) or taking a majority vote (for classification).
Key Parameters:
n_estimators=100: Specifies the number of decision trees in the forest. More trees generally improve performance but increase computation time.
class_weight='balanced': Adjusts weights for classes based on their frequencies, helping the model pay more attention to minority classes.
n_jobs=-1: Utilizes all available CPU cores for parallel processing, speeding up training.
random_state=42: Ensures reproducibility by setting a seed for random number generation.
Strengths:
Robust and handles both linear and non-linear data well.
Less prone to overfitting compared to individual decision trees due to averaging.
Can handle missing values and does not require feature scaling.
Weaknesses:
Can be computationally expensive, especially with many trees and a large dataset.
Less interpretable than single decision trees or logistic regression.
3. Gradient Boosting Classifier (gb_model)
Description:
Gradient Boosting builds an ensemble of decision trees in a sequential manner, where each new tree corrects the errors made by the previous ones. It optimizes a loss function using gradient descent.
Key Parameters:
n_estimators=100: Number of boosting stages (trees). More stages improve accuracy but increase computation time.
random_state=42: Ensures reproducibility.
Strengths:
High predictive power and often achieves better performance than Random Forest.
Handles complex, non-linear relationships well.
Effective for imbalanced data when combined with techniques like SMOTE.
Weaknesses:
Prone to overfitting if the number of trees (n_estimators) is too high.
Computationally intensive and slower to train than Random Forest.
4. Voting Classifier (ensemble_model)
Description:
The Voting Classifier is an ensemble meta-model that combines the predictions of multiple classifiers. In this case, we use Logistic Regression, Random Forest, and Gradient Boosting.
Voting Types:
Hard Voting: Takes the majority class label predicted by the individual models.
Soft Voting: Averages the predicted probabilities from each model and selects the class with the highest average probability.
Key Parameters:
estimators=[('lr', lr_model), ('rf', rf_model), ('gb', gb_model)]: Specifies the classifiers to be included in the ensemble.
voting='soft': Uses soft voting, which considers the confidence (probability) of each model’s prediction, often leading to better performance.
Why Use an Ensemble Model?
Combines Strengths:

Logistic Regression is strong for linearly separable data.
Random Forest handles non-linear data and reduces overfitting.
Gradient Boosting captures complex patterns and corrects errors sequentially.
Reduces Risk of Overfitting:

By combining multiple models, the ensemble reduces the likelihood that any single model’s error will dominate.
Improves Robustness and Accuracy:

The ensemble model leverages diverse decision-making processes, resulting in better generalization.
Training and Evaluation Process:
Before SMOTE:
The ensemble model is trained on the original dataset.
The performance metrics (precision, recall, F1-score) are calculated and analyzed.
After SMOTE:
The ensemble model is retrained on the resampled dataset (balanced using SMOTE).
The performance metrics are recalculated to observe the impact of SMOTE.
Why Use SMOTE?
SMOTE (Synthetic Minority Over-sampling Technique) helps balance the dataset by generating synthetic samples for the minority classes. This is crucial because:
It prevents the model from being biased towards majority classes.
It improves recall and F1-score for minority classes, making the model more robust and fair.
Summary of Approach:
Preprocessing: Cleaned text data using lemmatization and stopword removal.
Vectorization: Used TF-IDF with bigrams to capture important terms and phrases.
Balancing: Applied SMOTE to handle class imbalance.
Modeling: Built a robust ensemble model combining Logistic Regression, Random Forest, and Gradient Boosting.
Evaluation: Compared performance before and after SMOTE using precision, recall, and F1-score.
This ensemble approach leverages the strengths of multiple classifiers and handles both linearly separable and non-linear data effectively, leading to high accuracy and balanced performance across classes.