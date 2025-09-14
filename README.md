# Loan Approval Prediction System

This project is a **Loan Approval Prediction System** built using **Machine Learning (ML)** and **Deep Learning (DL)**.
It predicts whether a loan application will be **Approved (0)** or **Not Approved (1)** based on applicant information such as age, income, employment type, credit score, and more.

---

## 🚀 Features

* **Frontend with Streamlit**: Simple and interactive UI for user inputs.
* **Machine Learning Model**: Achieved **82% accuracy**.
* **Deep Learning Model**: Achieved **87% accuracy** with a neural network (Sigmoid activation at output).
* **End-to-End Pipeline**: Includes preprocessing (OneHotEncoding, scaling, handling categorical features) and prediction.

---

## 📊 Dataset Features

The system uses the following input features:

* **Age** – Applicant’s age
* **Income** – Monthly/annual income
* **LoanAmount** – Requested loan amount
* **CreditScore** – Creditworthiness score
* **MonthsEmployed** – Total months employed
* **NumCreditLines** – Number of existing credit lines
* **InterestRate** – Interest rate for loan
* **LoanTerm** – Duration of loan in months/years
* **DTIRatio** – Debt-to-Income ratio
* **Education** – Applicant’s education level
* **EmploymentType** – Type of employment (Salaried, Self-Employed, etc.)
* **MaritalStatus** – Single/Married/Divorced
* **HasMortgage** – Whether applicant has an existing mortgage
* **HasDependents** – Whether applicant has dependents
* **LoanPurpose** – Purpose of loan (Personal, Home, Auto, Education, etc.)
* **HasCoSigner** – Whether loan has a co-signer

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow/Keras** (for DL model)
* **Pandas, NumPy** (data handling)
* **Streamlit** (frontend UI)

---

## 📈 Model Performance

| Model Type       | Accuracy       |
| ---------------- | -------------- |
| Machine Learning | **0.82 (82%)** |
| Deep Learning    | **0.87 (87%)** |

---
## 📌 Usage

* Enter applicant details in the form.
* Click **Predict**.
* Get output:

  * **0 → Loan Approved**
  * **1 → Loan Not Approved**

