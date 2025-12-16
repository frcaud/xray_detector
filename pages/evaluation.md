# Evaluation

## Evaluation Metrics

The performance of submitted models is evaluated using two complementary regression metrics:

### Mean Squared Error (MSE)

The primary evaluation metric is the **Mean Squared Error (MSE)**, which measures the average squared difference between the predicted inflection points and the true inflection points.

### R² Score (Coefficient of Determination)

The **R² score** (also known as the coefficient of determination) measures the proportion of variance in the true inflection points that is explained by the model.

## Evaluation Procedure

The evaluation follows a standard machine learning competition workflow:

1. **Model Initialization**: For each evaluation, the model is initialized. You can also use pre-trained models.

2. **Training**: 
   - The model receives training data (`X_train`, `y_train`) and domain adaptation data (`X_adapt`)
   - The model's `fit()` method is called with these inputs

3. **Prediction**:
   - The model receives test data (`X_test`) without labels
   - The model's `predict()` method is called to generate predictions
   - Predictions are saved to the output directory

4. **Scoring**:
   - Predictions are compared against ground truth test labels
   - MSE and R² scores are computed
   - Execution time (duration) is also recorded

5. **Leaderboard Ranking**: 
   - Models are ranked primarily by **MSE** (lower is better)
   - R² score and execution time are provided as additional metrics for reference

## Submitting a solution to Codabench

Edit the submission file 'model.py' without changing the file name and compress it into a zip file.
Now submit the zip file to Codabench in 'My Submissions' tab.

You can make use of the GPU that is available on compute workers.

You can also review the logs on each submission to see where errors occur.