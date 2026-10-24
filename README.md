# Medical-Insurance-Cost
Predicting how much medical insurance cost a person will.
# Medical Insurance Charges Prediction using Linear Regression

## Overview

This project uses a medical insurance dataset to predict the **insurance
charges** a person may be charged based on health, lifestyle, and
financial details.

The project is implemented in a Python/Jupyter Notebook and demonstrates
a complete basic machine-learning workflow:

-   Loading and inspecting the dataset
-   Exploratory data analysis
-   Outlier removal using the IQR method
-   Encoding categorical variables
-   Splitting data into training and testing sets
-   Training a Linear Regression model
-   Generating predictions
-   Evaluating the model using R² score
-   Visualizing actual vs. predicted charges

The notebook starts with **1,338 rows and 13 columns** and, after the
preprocessing/outlier-removal steps shown in the notebook, the modeling
dataset contains **851 rows**.

## Dataset

The notebook uses a medical insurance dataset containing the following
columns:

  Feature                             Description
  ----------------------------------- ------------------------------------
  `age`                               Age of the person
  `sex`                               Sex
  `bmi`                               Body Mass Index
  `children`                          Number of children
  `smoker`                            Smoking status
  `Claim_Amount`                      Claim amount
  `past_consultations`                Number of past consultations
  `num_of_steps`                      Number of steps
  `Hospital_expenditure`              Hospital expenditure
  `NUmber_of_past_hospitalizations`   Number of past hospitalizations
  `Anual_Salary`                      Annual salary
  `region`                            Region
  `charges`                           Target variable: insurance charges

Dataset link provided in the notebook:

https://drive.google.com/file/d/1ge3HonOmUjtgEIHpUrtlnCl9N4ALYFbO/view?usp=sharing

## Technologies and Libraries

The notebook uses:

-   Python 3
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn

Key Scikit-learn components used include:

-   `LabelEncoder`
-   `LinearRegression`
-   `train_test_split`
-   `r2_score`

## Project Workflow

### 1. Load the data

The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`.

### 2. Inspect the data

The notebook examines the dataset and its columns. The original dataset
contains 13 columns, with `charges` used as the prediction target.

### 3. Outlier handling

The notebook applies the **Interquartile Range (IQR)** method to filter
outliers from:

-   `past_consultations`
-   `Hospital_expenditure`
-   `Anual_Salary`
-   `NUmber_of_past_hospitalizations`

After these filtering steps, the displayed modeling dataset contains 851
rows.

### 4. Encode categorical variables

The categorical columns are:

-   `sex`
-   `smoker`
-   `region`

These columns are converted to numeric values using `LabelEncoder`.

### 5. Define features and target

The target variable is:

``` python
y = df['charges']
```

The input features are created by removing `charges` from the DataFrame:

``` python
x = df.drop('charges', axis=1)
```

### 6. Train-test split

The data is split using:

``` python
train_test_split(x, y, train_size=0.80, random_state=0)
```

This produces:

-   **680 training rows**
-   **171 testing rows**

### 7. Train the Linear Regression model

A Linear Regression model is created and fitted:

``` python
model = LinearRegression()
model.fit(x_train, y_train)
```

### 8. Make predictions

The trained model is used to predict insurance charges for the test
data. The notebook also creates a result table containing:

-   Actual Values
-   Predicted Values

The result contains 171 test predictions.

### 9. Evaluate the model

The notebook evaluates the model using the R² score:

``` python
r2_score(y_test, predict) * 100
```

The recorded result is approximately:

**95.64% R² score**

### 10. Visualize predictions

A Seaborn regression plot is generated with:

-   X-axis: predicted charges values
-   Y-axis: actual charges values
-   Title: Regression Plot

## Results

The Linear Regression model achieved an R² score of approximately
**95.64%** on the test set used in the notebook.

This indicates that the fitted model explains a large proportion of the
variation in the observed insurance charges for this particular
train/test split.

## How to Run

1.  Open the notebook `Linear_Regression_Handson(14_02_2026).ipynb` in
    Jupyter Notebook, JupyterLab, or Google Colab.
2.  Make sure the required Python libraries are installed.
3.  Download or otherwise make the dataset available at the location
    expected by the notebook.
4.  Run the notebook cells from top to bottom.
5.  Review the preprocessing steps, model predictions, evaluation score,
    and regression plot.

Example installation command:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Project Structure

``` text
.
├── Linear_Regression_Handson(14_02_2026).ipynb
├── new_insurance_data (1) (2).csv
└── README.md
```

> The CSV filename above reflects the filename used by the notebook's
> `pd.read_csv()` call; ensure the actual dataset file is available in
> the expected path.

## Notes

-   This README is based on the workflow and recorded outputs present in
    the supplied notebook.
-   The notebook uses label encoding for categorical variables rather
    than one-hot encoding.
-   The reported R² value is the value recorded in the notebook for its
    test split.
-   The project is intended as a hands-on Linear Regression exercise for
    predicting medical insurance charges.
