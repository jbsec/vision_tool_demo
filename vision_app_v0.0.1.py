import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import plotly.graph_objs as go

# Define the parameters for the time series
hours = 48
minutes_per_hour = 60
total_minutes = hours * minutes_per_hour

# Time array
time_index = pd.date_range(start="2023-04-01 00:00", periods=total_minutes, freq='T')

# Simulating the features
np.random.seed(42)  # for reproducibility
total_calls = 50 + 30 * np.sin(np.linspace(0, 4 * np.pi, total_minutes)) + np.random.randint(0, 10, total_minutes)
dropped_calls = (total_calls * 0.05) + np.random.randint(0, 3, total_minutes)
average_call_duration = np.abs(np.random.normal(120, 30, total_minutes))  # absolute to avoid negative durations
peak_call_time = np.random.uniform(0, 60, total_minutes)
call_failures = (total_calls * 0.1) + np.random.randint(0, 5, total_minutes)
customer_complaints = np.random.poisson(0.1, total_minutes)  # very few complaints per minute

# Create the DataFrame
data = pd.DataFrame({
    'Dropped Calls': dropped_calls.astype(int),
    'Average Call Duration': average_call_duration.round(1),
    'Peak Call Time': peak_call_time.round(1),
    'Call Failures': call_failures.astype(int),
    'Customer Complaints': customer_complaints
})
X = data
y = total_calls.astype(int)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Explain model predictions using SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
base_value = explainer.expected_value

# Calculate the maximum SHAP value for consistent scaling across all instances
max_shap_value = np.max(np.abs(shap_values.values))

# Streamlit app
st.title('SHAP Value Visualizer')

# Slider for selecting the instance to visualize
selected_instance_index = st.slider('Select instance', 0, len(X_test)-1, 0)

# Features to display
features_to_display = ['Dropped Calls', 'Average Call Duration', 'Peak Call Time', 'Call Failures', 'Customer Complaints']
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different colors for each feature

# Extracting the SHAP values for the selected instance
instance_shap_values = shap_values.values[selected_instance_index]

# Create the plotly figure
fig = go.Figure()

# Add bars for each SHAP value of the selected instance
for i, feature in enumerate(features_to_display):
    fig.add_trace(go.Bar(
        x=[feature], 
        y=[instance_shap_values[X_test.columns.get_loc(feature)]],
        name=feature, 
        marker_color=colors[i]
    ))

# Add the base value for reference
fig.add_trace(go.Scatter(
    x=[features_to_display[-1]], 
    y=[base_value], 
    mode='markers+text',
    text=['Base Value'],
    textposition='bottom center',
    marker=dict(color='grey', size=10),
    showlegend=False
))

# Add a marker for the predicted value
predicted_value = base_value + instance_shap_values.sum()
fig.add_trace(go.Scatter(
    x=[features_to_display[-1]], 
    y=[predicted_value],
    mode='markers+text',
    text=['Predicted Value'],
    textposition='top center',
    marker=dict(color='black', size=10),
    showlegend=False
))

# Set the figure layout
fig.update_layout(
    title=f'SHAP Values for Instance {selected_instance_index}',
    xaxis_title='Feature',
    yaxis_title='SHAP Value',
    yaxis=dict(range=[-max_shap_value, max_shap_value * 1.10]),
    barmode='group'
)

# Display the figure in the Streamlit app
st.plotly_chart(fig, use_container_width=True)
