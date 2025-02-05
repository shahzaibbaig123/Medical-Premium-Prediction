# Medical Premium Prediction Analysis

## Overview
This project uses machine learning techniques to predict medical premium prices based on a variety of health-related factors. The analysis is performed on a dataset that includes information on patient demographics, medical history, and existing health conditions. This dataset is taken from Kaggle. Various predictive models, including regression trees, random forests, and linear stepwise regression, are implemented to assess the most effective method for predicting premium prices.

## Project Structure

### 1. **Data Preparation**
   - The dataset is loaded and cleaned to handle missing values and standardize column names.
   - Variables such as `diabetes`, `blood_pressure_problems`, `any_transplants`, and others are transformed into categorical factors for further analysis.

### 2. **Exploratory Data Analysis (EDA)**
   - Boxplots and scatter plots are used to visualize the relationships between medical conditions and premium prices.
   - Histograms are also plotted to understand the distribution of premium prices.

### 3. **Modeling**
   - **Linear Regression**: Stepwise forward and backward selection methods are used to build predictive models.
   - **Regression Trees**: A decision tree model is built to predict the premium price based on the provided features.
   - **Random Forest**: A random forest model is built to assess the importance of various features in predicting the premium price.
   
### 4. **Model Evaluation**
   - The models' performance is evaluated using R-squared and Mean Squared Error (MSE) on both the training and testing datasets.
   - The best-performing model is selected based on MSE and R-squared comparisons across all models.

## Requirements
Ensure the following libraries are installed before running the analysis:
- `dplyr`
- `janitor`
- `tidyverse`
- `ggplot2`
- `readr`
- `gridExtra`
- `rsample`
- `randomForest`
- `rpart`
- `rpart.plot`
- `gbm`
- `glmnet`
- `tree`
- `RColorBrewer`
- `pls`
- `jtools`

## How to Run
1. Install the necessary libraries:
    ```R
    install.packages(c("dplyr", "janitor", "tidyverse", "ggplot2", "readr", "gridExtra", "rsample", "randomForest", "rpart", "rpart.plot", "gbm", "glmnet", "tree", "RColorBrewer", "pls", "jtools"))
    ```

2. Load the dataset and libraries in your R environment:
    ```R
    library(dplyr)
    library(tidyverse)
    # Add all other libraries similarly
    Medicalpremium <- read_csv("path/to/Medicalpremium.csv")
    ```

3. Follow the steps outlined in the script to explore the data, build models, and evaluate them.
