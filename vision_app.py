import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Set the page to wide mode
st.set_page_config(layout="wide")

# Simulated data parameters
hours = 48
minutes_per_hour = 60
total_minutes = hours * minutes_per_hour
time_index = pd.date_range(start="2023-04-01 00:00", periods=total_minutes, freq='T')
np.random.seed(90210)
total_calls = 50 + 30 * np.sin(np.linspace(0, 4 * np.pi, total_minutes)) + np.random.randint(0, 10, total_minutes)
dropped_calls = (total_calls * 0.05) + np.random.randint(0, 3, total_minutes)
average_call_duration = np.abs(np.random.normal(120, 30, total_minutes))
peak_call_time = np.random.uniform(0, 60, total_minutes)
call_failures = (total_calls * 0.1) + np.random.randint(0, 5, total_minutes)
customer_complaints = np.random.poisson(0.1, total_minutes)
data = pd.DataFrame({
    'Dropped Calls': dropped_calls.astype(int),
    'Average Call Duration': average_call_duration.round(1),
    'Peak Call Time': peak_call_time.round(1),
    'Call Failures': call_failures.astype(int),
    'Customer Complaints': customer_complaints
})
X = data
y = total_calls.astype(int)

# Model Caching (Makes the app faster)
@st.cache(allow_output_mutation=True)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model(X, y)

# SHAP values computation
@st.cache(allow_output_mutation=True)
def get_shap_values(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    return shap_values, explainer.expected_value

shap_values, base_value = get_shap_values(model, X_test)

# Create three equally wide columns
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader('Time Series Analysis of Call Data Features')
    fig_ts = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=['Dropped Calls', 'Average Call Duration', 'Peak Call Time', 'Call Failures', 'Customer Complaints'])
    for i, feature in enumerate(['Dropped Calls', 'Average Call Duration', 'Peak Call Time', 'Call Failures', 'Customer Complaints']):
        fig_ts.add_trace(
            go.Scatter(x=time_index, y=data[feature], mode='lines'),
            row=i + 1, col=1
        )
    for i in range(1, 6):
        fig_ts.update_xaxes(title_text="Time", row=i, col=1)
    fig_ts.update_layout(height=600, title_text="Feature Trends Over Time", showlegend=False)
    st.plotly_chart(fig_ts, use_container_width=True)
    
with col2:
    st.title('Feature Impact View')
    if shap_values is not None:
        selected_instance_index = st.slider('Select instance', 0, len(X_test) - 1, 0)
        features_to_display = ['Dropped Calls', 'Average Call Duration', 'Peak Call Time', 'Call Failures', 'Customer Complaints']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        instance_shap_values = shap_values.values[selected_instance_index]
        max_shap_value = np.max(np.abs(shap_values.values)) if len(shap_values.values) > 0 else 0
        fig = go.Figure()
        for i, feature in enumerate(features_to_display):
            fig.add_trace(go.Bar(
                x=[feature],
                y=[instance_shap_values[X_test.columns.get_loc(feature)]],
                name=feature,
                marker_color=colors[i]
            ))
        predicted_value = base_value + instance_shap_values.sum()
        fig.add_trace(go.Scatter(
            x=[features_to_display[-1]], 
            y=[predicted_value],
            mode='markers+text',
            text=['Predicted Value'],
            textposition='top center',
            marker=dict(color='black', size=12),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[features_to_display[0]], 
            y=[base_value],
            mode='markers+text',
            text=['Base Value'],
            textposition='bottom center',
            marker=dict(color='grey', size=12),
            showlegend=False
        ))
        fig.update_layout(
            title=f'SHAP Values for Instance {selected_instance_index}',
            xaxis_title='Feature',
            yaxis_title='SHAP Value Impact',
            yaxis=dict(range=[-max_shap_value, max_shap_value * 1.1]),
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        description = """**Interpretation Guide:** This visualization shows the SHAP values for the selected model instance. Each bar represents the impact of a feature on the model's prediction. Positive values increase the prediction, while negative values decrease it. This insight helps understand which features are most influential for specific predictions and why the model behaves as it does in certain scenarios. """
        st.text_area("The Value of This XAI", value=description, height=150)
    else:
        st.error("SHAP values could not be computed. Please check your model and input data.")

with col3:
    st.subheader("Sankey Diagram of Feature Contributions")
    if shap_values is not None:
        # Sankey diagram setup
        feature_names = X_test.columns.tolist()
        selected_instance_shap_values = shap_values.values[selected_instance_index]
        
        # Define nodes for Sankey diagram
        nodes = feature_names + ['Total Calls']
        nodes_indices = list(range(len(nodes)))  # Generate a list of node indices

        # Define source, target, and value for flows
        source = nodes_indices[:-1]  # All feature indices
        target = [len(nodes_indices) - 1] * len(feature_names)  # Target is the 'Total Calls' node for all features
        value = selected_instance_shap_values.tolist()

        # Define link colors by their SHAP value (blue for positive, red for negative)
        link_color = [
            'rgba(0, 0, 255, 0.5)' if val > 0 else 'rgba(255, 0, 0, 0.5)'
            for val in value
        ]

        # Create figure
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes
            ),
            link=dict(
                source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                target=target,
                value=value,
                color=link_color
            ))])

        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.write("Select an instance to generate the Sankey diagram.")

# ... (any additional Streamlit code you want to include)
