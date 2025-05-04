# Employee Productivity Analysis

**Objective:**  
This project provides an **interactive**, **visual** and **predictive** analysis of employee productivity and performance, comparing metrics **before** and **after** leave periods, to help managers identify trends and anticipate workforce changes.

---

## 🛠 **Tools Used**
- **Python**  
  - Flask  
  - pandas  
  - numpy  
  - scikit‑learn  
  - joblib  
  - Plotly  
- **HTML/CSS**  
- **JavaScript (vanilla)**

---

## 📊 **Dataset Attribution**

The data comes from internal HR records (anonymized) and has been pre‑processed for modeling:

- **Raw data** is located in `data/raw/`
- **Cleaned data** is in `data/cleaned/` (`Cleaned_Data.csv`)

---

## 🔍 **Key Features and Analysis Steps**

1. **Data Loading & Preprocessing**  
   - Read `Cleaned_Data.csv`  
   - Compute **tenure**, **leaves taken**, and **projects per year**  

2. **Exploratory Analysis**  
   - Summary statistics of productivity and performance  
   - Visualize distributions (histograms, box plots)  
   - Time‑series plots to spot trends  

3. **Interactive Dashboard**  
   - Form to input an **Employee ID**  
   - Display employee details:
     - Hiring date
     - Tenure
     - Leaves taken
     - Projects completed per year  
   - Plotly charts:
     - **Bar chart**: Productivity before vs. after leave  
     - **Bar chart**: Performance before vs. after leave  
     - **Pie chart**: Time allocation (sick days, overtime, regular work)

4. **Predictive Modeling**  
   - Load pre‑trained model:  
     - **Productivity & Performance Predictor** (`models/employee_productivity_model.pkl`)
   - Input features:
     - Likes → Tenure, Leaves, Projects/year (numeric)  
   - Output:
     - Predicted scores for productivity and performance (before & after leave)

5. **Export Results**  
   - Option to download predictions as CSV  
   - Clear UI feedback during download  

---

## 🧠 **Predictive Model Details**

- **Employee Productivity & Performance Model** (`models/employee_productivity_model.pkl`)  
  - Trained on features:  
    - Tenure (years)  
    - Leaves taken (count)  
    - Projects completed per year  
  - Outputs four scores:  
    - Productivity before leave  
    - Productivity after leave  
    - Performance before leave  
    - Performance after leave  

---

For deeper analysis, see `Analysis.ipynb`.

---

© 2025 Girish Shenoy
