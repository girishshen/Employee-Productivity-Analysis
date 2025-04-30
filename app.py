from flask import Flask, render_template
import pandas as pd
import numpy as np
import plotly.express as px

app = Flask(__name__)

# ─── Chart Generators ─────────────────────────────────────────────────────

def generate_performance_boxplot(df):
    fig = px.box(
        df,y="Performance_Score",
        title="Performance Score Distribution",
        points="all",notched=True
    )

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Performance Score")
    return fig.to_html(full_html=False)


def generate_top_feature_correlations(df, n=5):
    num_df = df.select_dtypes(include=[np.number])
    corr_series = num_df.corr()["Performance_Score"].drop("Performance_Score")
    top_feats = corr_series.abs().nlargest(n).index
    top_corrs = corr_series.loc[top_feats]

    fig = px.bar(
        x=top_corrs.index, y=top_corrs.values,
        title=f"Top {n} Feature Correlations with Performance Score",
        labels={"x": "Feature", "y": "Correlation"}
    )

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_tickformat=".2f")
    return fig.to_html(full_html=False)


def generate_training_vs_performance_boxplot(df):
    fig = px.box(
        df, x="Performance_Score", y="Training_Hours",
        title="Training Hours by Performance Score",
        points="all"
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Performance Score", yaxis_title="Training Hours"
    )

    return fig.to_html(full_html=False)


def generate_productivity_histogram(df):
    fig = px.histogram(
        df, x="Productivity_Score", nbins=30,
        title="Distribution of Productivity Score"
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Productivity Score", yaxis_title="Count"
    )

    return fig.to_html(full_html=False)


def generate_training_performance_joint(df):
    fig = px.scatter(
        df, x="Training_Hours", y="Performance_Score",
        title="Training vs Performance with Regression",
        trendline="ols", marginal_x="histogram", marginal_y="histogram"
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Training Hours", yaxis_title="Performance Score"
    )

    return fig.to_html(full_html=False)


# ─── Main Route ─────────────────────────────────────────────────────────────

@app.route('/')
def index():

    # Load and preprocess data
    df = pd.read_csv('data/cleaned/Cleaned_Data.csv')
    # df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])

    df['Department'] = df['Department'].replace({
        0: "IT", 1: "Finance", 2: "Customer Support", 3: "Engineering",
        4: "Marketing", 5: "HR", 6: "Operation", 7: "Sales", 8: "Legal"
    })

    # Department performance table
    dept_df = df.groupby('Department')['Performance_Score'].mean().reset_index()
    department_performance = dept_df.to_dict(orient='records')

    # Insights for display
    insights = {
        "top_performers": (
            "Employees with training hours above the 75th percentile, balanced workloads (3–5 projects/year), "
            "and Engagement_Index > 1.5 show the highest average Performance Score."
        ),

        "underperforming_departments": (
            "Customer Support and HR average below 3.0; these teams log >10 overtime hours/week "
            "and have Satisfaction_Score < 3.5."
        ),

        "burnout_indicators": (
            "Teams working >12 overtime hours weekly with Engagement_Index < 1.0 are at high risk of burnout "
            "and performance decline."
        )
    }

    # Generate charts
    performance_chart      = generate_performance_boxplot(df)
    feature_corr_chart     = generate_top_feature_correlations(df, n=5)
    training_boxplot_chart = generate_training_vs_performance_boxplot(df)
    productivity_histogram = generate_productivity_histogram(df)
    joint_plot             = generate_training_performance_joint(df)

    # Render template
    return render_template(
        'dashboard.html',
        insights=insights,
        department_performance=department_performance,
        performance_chart=performance_chart,
        correlation_heatmap=feature_corr_chart,
        training_boxplot=training_boxplot_chart,
        productivity_hist=productivity_histogram,
        joint_plot=joint_plot
    )

if __name__ == '__main__':
    app.run(debug=True)