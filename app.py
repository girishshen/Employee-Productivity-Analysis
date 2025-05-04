from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model and dataset
model = joblib.load('models/employee_productivity_model.pkl')
data = pd.read_csv('data/cleaned/Cleaned_Data.csv')

# Get the feature names the model expects
# print("Model expected features: ", model.feature_names_in_)

def format_date_with_ordinal(date_str):
    # Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")

    # Get the day and apply ordinal suffix
    day = date_obj.day
    if 10 <= day <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

    # Format date as "1st May 2016"
    formatted_date = date_obj.strftime(f"%d{suffix} %B %Y")

    return formatted_date


# Ensure the columns match between training and prediction
feature_columns = [
    'Age', 'Monthly_Salary', 'Work_Hours_Per_Week',
    'Projects_Handled', 'Overtime_Hours', 'Sick_Days',
    'Remote_Work_Frequency', 'Team_Size', 'Training_Hours',
    'Promotions', 'Employee_Satisfaction_Score', 'Resigned',
    'Department_Engineering', 'Department_Finance', 'Department_HR',
    'Department_IT', 'Department_Legal', 'Department_Marketing',
    'Department_Operations', 'Department_Sales', 'Gender_Male',
    'Gender_Other', 'Job_Title_Consultant', 'Job_Title_Developer',
    'Job_Title_Engineer', 'Job_Title_Manager', 'Job_Title_Specialist',
    'Job_Title_Technician', 'Experience_Salary_Interaction', 'Workload_Intensity',
    'Rolling_Avg_Performance', 'Overtime_Work_Ratio', 'SickDays_WorkDays_Ratio'
]


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/predict', methods=['POST'])
def predict():
    emp_id = request.form['emp_id']
    employee = data[data['Employee_ID'] == int(emp_id)]

    if employee.empty:
        return render_template(
            'dashboard.html',
            error="Employee with ID {} not found.".format(emp_id)
        )

    # List of expected feature columns (36 features based on your model)
    expected_columns = [
        'Age', 'Monthly_Salary', 'Work_Hours_Per_Week',
        'Projects_Handled', 'Overtime_Hours', 'Sick_Days',
        'Remote_Work_Frequency', 'Team_Size', 'Training_Hours',
        'Promotions', 'Employee_Satisfaction_Score', 'Resigned',
        'Department_Engineering', 'Department_Finance', 'Department_HR',
        'Department_IT', 'Department_Legal', 'Department_Marketing',
        'Department_Operations', 'Department_Sales', 'Gender_Male',
        'Gender_Other', 'Job_Title_Consultant', 'Job_Title_Developer',
        'Job_Title_Engineer', 'Job_Title_Manager', 'Job_Title_Specialist',
        'Job_Title_Technician', 'Experience_Salary_Interaction', 'Workload_Intensity',
        'Rolling_Avg_Performance', 'Overtime_Work_Ratio', 'SickDays_WorkDays_Ratio',
        'Years_At_Company', 'Sick_Days', 'Projects_Handled'
    ]

    # Check for missing columns
    missing_columns = [col for col in expected_columns if col not in employee.columns]
    if missing_columns:
        return render_template(
            'dashboard.html',
            error="Missing columns for prediction: " + ", ".join(missing_columns))

    # Extract employee details
    hiring_date = format_date_with_ordinal(employee['Hire_Date'].iloc[0])
    tenure = round(employee['Years_At_Company'].iloc[0], 2)
    leaves = round(employee['Sick_Days'].iloc[0] / 365, 2)
    projects_per_year = round(employee['Projects_Handled'].iloc[0], 2)

    # Clean the data before prediction (replace Inf or NaN with 0)
    features = employee[expected_columns].values.reshape(1, -1)

    # Replace any infinities with NaN, then fill NaN values with 0
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    try:
        # Predict productivity performance
        prediction = model.predict(features)[0]

        # Simulate before-leave and after-leave performance
        before_leave = round(prediction + np.random.uniform(-0.05, 0.05), 2)
        after_leave = round(prediction + np.random.uniform(-0.05, 0.05), 2)

        # Generate performance trend chart (simulated data for demonstration)
        months = pd.date_range(start=hiring_date, periods=12, freq='ME')

        # Replace with actual performance data
        performance_scores = np.round(np.random.uniform(0.6, 0.8, size=12), 3)

        trace = go.Scatter(
            x=months,
            y=performance_scores,
            mode='lines+markers',
            name='Performance Trend',
            text=[f"{score:.3f}" for score in performance_scores],
            hoverinfo='text+x'
        )

        layout = go.Layout(
            title='Performance Trend Over Time',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Performance Score')
        )

        line_chart = pyo.plot(go.Figure(data=[trace], layout=layout), output_type='div')

        # Bar chart (productivity & performance before/after leave)
        trace_bar = go.Bar(
            x=['Before Leave', 'After Leave'],
            y=[before_leave, after_leave],
            marker_color=['#1f77b4', '#2ca02c']
        )

        layout_bar = go.Layout(title='Predicted Productivity & Performance', yaxis=dict(title='Score'))
        performance_chart = pyo.plot(go.Figure(data=[trace_bar], layout=layout_bar), output_type='div')

        return render_template(
            'dashboard.html',
            emp_id=emp_id,
            hiring_date=hiring_date,
            tenure=tenure,
            leaves=leaves,
            projects_per_year=projects_per_year,
            performance_chart=performance_chart,
            line_chart=line_chart
        )

    except Exception as e:
        return render_template(
            'dashboard.html',
            error=f"An error occurred: {e}"
        )


if __name__ == '__main__':
    app.run(debug=True)
