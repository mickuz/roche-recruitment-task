# Titanic - data science recruitment task project

The goal of this repository is to provide an example of a machine learning project going through fundamental steps such as: data exploration and analysis, feature engineering, building a prediction model, tuning hyperparameters and making predictions. [Titanic dataset](https://www.kaggle.com/c/titanic) was used for this project:

>The sinking of the Titanic is one of the most infamous shipwrecks in history.

>On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

>While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

>In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## Installation:

To run this project:
1. Clone the repository into your machine: `git clone https://github.com/mickuz/roche-recruitment-task.git`
2. Build an image from a Dockerfile: `docker build -t <image-name> .`
3. Run a container from an image interactively: `docker run -it <image-name>`
4. Execute the command `python3 train.py` to train the model
5. Execute the command `python3 predict.py` to make predictions and display the final performance of the model

## Table of Contents

**Data analysis:**
* Descriptive Analysis
* Visualizations
* Finding correlations between features

**Feature engineering:**
* Handling missing data
* Converting features from categorical to numerical
* One-hot encoding

**Predictive modeling:**
* Gaussian Naive Bayes
* Logistic Regression
* K-Nearest Neighbors
* Support Vector Machines
* Decision Tree
* Random Forest

**Evaluation of the model:**
* K-folds cross validation
* Accuracy and F1 scores