# README

This repository contains code for a classification task on the Haberman's Survival dataset. The dataset contains information about the survival of patients who had undergone surgery for breast cancer.

## Dataset

The dataset is loaded from the `haberman.data` file. It has the following columns:
- `age_at_operation`: Age of the patient at the time of the operation (numerical)
- `year_of_operation`: Year of the operation (numerical)
- `no_of_pos_aux_nodes`: Number of positive axillary nodes detected (numerical)
- `survival_status`: Survival status of the patient (1 - the patient survived 5 years or longer, 2 - the patient died within 5 years)

## Model Selection

The code performs model selection by implementing the k-nearest neighbors (KNN) algorithm using Euclidean distance as the distance metric. The steps involved are as follows:

1. Load the dataset into a Pandas DataFrame.
2. Normalize the data using min-max scaling.
3. Implement the KNN function that takes a DataFrame, K value, and a new datapoint to predict the class of the new datapoint.
4. Perform Leave-One-Out (LOO) cross-validation to find the optimal K value. LOO is applied for K values ranging from 1 to 99 with a step size of 2. The error rate for each K value is calculated.
5. Plot the error rates for different K values.
6. Find the optimal K value with the minimum error.
7. Plot the decision boundary by scattering points on a 2D grid and classifying them using the optimal K value.

## Estimating Parametric Models using MLE

The code also includes the implementation of parametric models using Maximum Likelihood Estimation (MLE). Three different covariance structures are considered: independent covariances, equal covariances, and diagonal covariances. The steps involved are as follows:

1. Implement classes for each covariance structure: `IndependentCovar`, `EqualCovar`, and `DiagonalCovar`.
2. Implement the `getCovar` method to calculate the covariance matrices for each class based on the training data.
3. Implement the `classify` method to classify a new datapoint using the MLE-based discriminant function for each covariance structure.
4. Perform model selection using LOO validation to find the optimal covariance structure for the dataset.
5. Calculate the test errors for each fold using the optimal covariance structure.
6. Calculate the average LOO validation error, average test error, and the optimal K value error.


```python
class IndependentCovar:
    def estimate_parameters(self, data):
        # Split the data based on the class labels
        C1 = data[data['survival_status'] == 1].drop('survival_status', axis=1)
        C2 = data[data['survival_status'] == 2].drop('survival_status', axis=1)

        # Calculate the prior probabilities
        P_C1 = len(C1) / len(data)
        P_C2 = len(C2) / len(data)

        # Calculate the mean vectors for each class
        m1 = C1.mean().values
        m2 = C2.mean().values

        # Calculate the covariance matrices for each class
        S1 = C1.cov().values
        S2 = C2.cov().values

        return P_C1, P_C2, m1, m2, S1, S2
```

we can use this `estimate_parameters` method to estimate the parameters of the Gaussian distribution for the IndependentCovar class.

Similarly, you can implement the `estimate_parameters` method for the EqualCovar and DiagonalCovar classes by replacing the code with the respective formulas for estimating parameters in those cases.

```
## Usage

1. Clone the repository.
2. Install the required dependencies (Pandas, NumPy, Matplotlib).
3. Run the code in a Python environment (e.g., Jupyter Notebook).

Note: Ensure that the `haberman.data` file is present in the same directory as the code.

## Results

The code provides the following results:
- The optimal K value for KNN.
- The decision boundary plot for the optimal K value.
- The average LOO validation error, average test error, and optimal K value error for the MLE-based models using different covariance structures.

Feel free to explore the code and experiment with different settings and configurations.
