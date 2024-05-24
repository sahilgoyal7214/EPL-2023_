# Premier League Table Prediction

## Introduction

In the realm of global football, the English Premier League stands out as a beacon of quality and competition. It boasts top-tier teams and is renowned for its unpredictability and passionate matches. This project aims to leverage data analysis to predict Premier League match outcomes, providing clubs with insights to gain a competitive edge. Our goal is to take advantage of this scenario and introduce a groundbreaking approach that gives Premier League clubs access to data’s predictive capabilities. We aim to enable these clubs to see beyond the short term and foresee the outcomes of their future matches against their fellow competitors. It’s an exploration of the core concepts in football analytics, blending historical performance analysis with the optimum statistical tools required for future projection.  

## Problem Statement

The main goal of this work is to create the most influential features through feature engineering to accurately determine the outcome in multi-class and binary-class of EPL matches. Our primary objective is to furnish specific clubs with valuable statistics and insights, allowing us to forecast the likelihood of their success in upcoming matches against opposing teams and generate a table using that outcome.

## Dataset Description

Data was scraped from various sources, including head-to-head match results, individual player statistics, team statistics, and player ratings. These datasets provide comprehensive insights into team and player performance over the past two seasons.

## Data Collection and Preprocessing

Steps were taken to clean and preprocess the data, including removing unwanted columns, handling missing data, correcting garbage values, creating new columns for additional insights, label encoding and standardizing team names.

## Exploratory Data Analysis (EDA)

EDA was conducted to gain insights into team performance, player statistics and match factors such as venue. Visualizations were used to analyze correlations and trends within the data.

## Model Testing and Building

Various machine learning models were tested, including Random Forest Classifier, SVM, GBM, KNN and Gaussian Naive Bayes. Random Forest Classifier emerged as the best-performing model, with high accuracy, precision, recall and Jaccard Score.

![image](https://github.com/saharshmehrotra/_premierleagueprediction/assets/135410807/ae391ca8-d5ed-42a8-ab5e-70513c7ebfc8)


## Model Deployment

The predictive model was deployed using Streamlit, creating a user-friendly web interface. Users can access live table predictions, team data and interactive EDA graphs.

![image](https://github.com/saharshmehrotra/_premierleagueprediction/assets/135410807/f41932ae-ba17-4867-b180-ed70c921c43a)

![image](https://github.com/saharshmehrotra/_premierleagueprediction/assets/135410807/50dbf0c1-6c32-4b82-91be-13d36f61d3c5)

![image](https://github.com/saharshmehrotra/_premierleagueprediction/assets/135410807/daff75f5-93b1-4b52-be7f-bd5d9c34433a)


## Conclusion

This project demonstrates the application of Python and machine learning techniques to predict Premier League match outcomes. While the model provides valuable insights, the unpredictable nature of football means that predicting winners remains a challenging task.

*UPDATE:* As the 2023/24 season of the Premier League has finally come to a conclusion, the results were quite impressive! Accurately predicting Manchester City as champions, with Arsenal and Liverpool finishing second and third, respectively. Despite the inherent unpredictability of sports, the model placed the top 10 teams within one or two spots of their actual positions and correctly predicted 2 out of 3 relegated teams✅️

This underscores the robustness of our approach and it is evident in the tabular comparison of actual vs. predicted positions given below:
<p align="center">
<img width="503" alt="PL Table - Actual vs Predicted 2023-24" src="https://github.com/saharshmehrotra/_premierleagueprediction/assets/135410807/c4ababc2-e816-4d36-b8be-ba757e7ef6f8">
</p>


## Deployment Link

[Premier League Table Prediction](https://premier-league.streamlit.app/)

---

*Note: For detailed implementation and code, refer to the respective Python scripts and notebooks.*
