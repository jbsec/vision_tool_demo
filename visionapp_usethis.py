import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Setting Streamlit layout (big screen mode)
st.set_page_config(layout="wide")

### Simulating some random data ###

hours = 48
minutes_per_hour = 60
total_minutes = hours * minutes_per_hour

time_index = pd.date_range(start="2023-04-01 00:00", periods=total_minutes, freq='T')

np.random.seed(90210)  # Fix randomness for consistent results

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


# Model caching (so the app doesn't retrain every time)
@st.cache(allow_output_mutation=True)
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Using random forest
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# Get the trained model
model, X_train, X_test, y_train, y_test = train_model(X, y)


# Compute SHAP values (helps explain predictions)
@st.cache(allow_output_mutation=True)
def get_shap_values(model, X_test):

    explainer = shap.Explainer(model)  # SHAP magic
    shap_values = explainer(X_test)  

    return shap_values, explainer.expected_value

shap_values, base_value = get_shap_values(model, X_test)


# --- Streamlit UI --- #

col1, col2, col3 = st.columns([1, 1, 1])  # Making 3 columns

# -------- Time Series Plot -------- #
with col1:
    st.subheader('Time Series Plots')

    fig_ts = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=['Dropped Calls', 'Average Call Duration', 'Peak Call Time', 'Call Failures', 'Customer Complaints'])

    for i, feature in enumerate(data.columns):  # Loop over features
        fig_ts.add_trace(
            go.Scatter(x=time_index, y=data[feature], mode='lines'),
            row=i + 1, col=1
        )

    for i in range(1, 6):
        fig_ts.update_xaxes(title_text="Time", row=i, col=1)

    fig_ts.update_layout(height=600, title_text="Feature Trends Over Time", showlegend=False)

    st.plotly_chart(fig_ts, use_container_width=True)


# -------- SHAP Feature Impact -------- #
with col2:
    st.title('Feature Impact View')

    if shap_values is not None:
        selected_instance_index = st.slider('Select instance', 0, len(X_test) - 1, 0)

        features_to_display = data.columns.tolist()
        colors = ['blue', 'green', 'red', 'purple', 'orange']  # Just picked some random colors

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

        description = """ SHAP values show how much each feature pushed the model’s prediction up or down. Bigger bars mean bigger influence. """
        st.text_area("Interpretation Guide", value=description, height=100)

    else:
        st.error("Oops... SHAP values couldn't be computed.")


# -------- Sankey Diagram -------- #
with col3:
    st.subheader("Sankey Diagram of Feature Contributions")

    if shap_values is not None:

        feature_names = X_test.columns.tolist()
        selected_instance_shap_values = shap_values.values[selected_instance_index]

        # Defining nodes
        nodes = feature_names + ['Total Calls']
        nodes_indices = list(range(len(nodes)))

        source = nodes_indices[:-1]  
        target = [len(nodes_indices) - 1] * len(feature_names)  # Connect all features to the final prediction node
        value = selected_instance_shap_values.tolist()

        # Color logic for Sankey links
        link_color = ['rgba(0, 0, 255, 0.5)' if val > 0 else 'rgba(255, 0, 0, 0.5)' for val in value]

        # Build Sankey diagram
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_color
            ))])

        st.plotly_chart(fig_sankey, use_container_width=True)

    else:
        st.write("No instance selected.")


# ... (Random other Streamlit stuff you might want to add)
