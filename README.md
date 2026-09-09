# 🚢 Titanic Survival Prediction

## 📌 Project Overview

Titanic Survival Prediction is a Machine Learning project that predicts whether a passenger survived the Titanic disaster based on various passenger details such as passenger class, gender, age, family information, fare, and port of embarkation.

The project demonstrates the complete Machine Learning workflow, including data exploration, data preprocessing, feature selection, model training, prediction, and evaluation.

---

## 🎯 Objective

The main objective of this project is to build a Machine Learning model that can predict passenger survival on the Titanic dataset.

The model learns patterns from historical passenger data and predicts one of two outcomes:

- `0` → Did not survive
- `1` → Survived

---

## 📊 Dataset

The project uses the **Titanic Dataset**, which contains information about 891 passengers.

### Dataset Features

| Feature | Description |
|---|---|
| PassengerId | Unique passenger identification number |
| Survived | Survival status (0 = No, 1 = Yes) |
| Pclass | Passenger class (1st, 2nd, or 3rd) |
| Name | Passenger's name |
| Sex | Passenger's gender |
| Age | Passenger's age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation |

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Examined the structure and statistical information of the dataset.
- Identified missing values.
- Handled missing values in the `Age` and `Embarked` columns.
- Removed unnecessary columns such as `PassengerId`, `Name`, `Ticket`, and `Cabin`.
- Converted categorical variables such as `Sex` and `Embarked` into numerical values.
- Prepared the cleaned dataset for Machine Learning.

---

## 🤖 Machine Learning

The processed dataset was divided into training and testing sets.

The model was trained using passenger characteristics to predict the `Survived` target variable.

### Target Variable

```text
Survived
