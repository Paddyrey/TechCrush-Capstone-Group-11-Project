# Import Libraries

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# Page Configuration

st.set_page_config(
    page_title="Household Energy AI",
    page_icon="⚡",
    layout="wide"
)


# Custom Styling

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 28px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #dbeafe;
        max-width: 850px;
    }

    .metric-card {
        background-color: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        height: 145px;
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 10px;
    }

    .section-card {
        background-color: white;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-top: 18px;
    }

    .recommendation-card {
        background-color: #f8fafc;
        padding: 16px;
        border-radius: 14px;
        border-left: 6px solid #2563eb;
        margin-bottom: 12px;
    }

    .warning-card {
        background-color: #fff7ed;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #f97316;
        margin-top: 14px;
    }

    .safe-card {
        background-color: #ecfdf5;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #10b981;
        margin-top: 14px;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# File Paths

DATA_DIR = Path("data")
MODEL_DIR = Path("models")


# Load Saved Models and Files

@st.cache_resource
def load_models():
    prediction_model = joblib.load(MODEL_DIR / "energy_consumption_prediction_model.pkl")
    model_features = joblib.load(MODEL_DIR / "energy_model_features.pkl")
    category_thresholds = joblib.load(MODEL_DIR / "consumption_category_thresholds.pkl")

    kmeans_model = joblib.load(MODEL_DIR / "energy_kmeans_model.pkl")
    cluster_scaler = joblib.load(MODEL_DIR / "energy_cluster_scaler.pkl")
    clustering_features = joblib.load(MODEL_DIR / "energy_clustering_features.pkl")
    cluster_segment_mapping = joblib.load(MODEL_DIR / "cluster_segment_mapping.pkl")

    anomaly_model = joblib.load(MODEL_DIR / "energy_anomaly_model.pkl")
    anomaly_scaler = joblib.load(MODEL_DIR / "energy_anomaly_scaler.pkl")
    anomaly_features = joblib.load(MODEL_DIR / "energy_anomaly_features.pkl")
    anomaly_thresholds = joblib.load(MODEL_DIR / "anomaly_thresholds.pkl")

    return {
        "prediction_model": prediction_model,
        "model_features": model_features,
        "category_thresholds": category_thresholds,
        "kmeans_model": kmeans_model,
        "cluster_scaler": cluster_scaler,
        "clustering_features": clustering_features,
        "cluster_segment_mapping": cluster_segment_mapping,
        "anomaly_model": anomaly_model,
        "anomaly_scaler": anomaly_scaler,
        "anomaly_features": anomaly_features,
        "anomaly_thresholds": anomaly_thresholds
    }


@st.cache_data
def load_dataset():
    file_path = DATA_DIR / "processed" / "energy_household_with_anomalies.csv"

    if file_path.exists():
        return pd.read_csv(file_path)

    return None


try:
    saved_items = load_models()
    energy_df = load_dataset()

except FileNotFoundError as e:
    st.error("Some required model files are missing. Please complete the notebook stages before running the app.")
    st.write(e)
    st.stop()


# Helper Functions

def band_to_supply_hours(service_band):
    band_hours = {
        "Band A": 20,
        "Band B": 16,
        "Band C": 12,
        "Band D": 8,
        "Band E": 4,
        "Below Band E / Low Supply": 2
    }

    return band_hours.get(service_band, 12)


def create_user_profile(
    household_size,
    number_of_rooms,
    service_band,
    light_bulb_count,
    fan_count,
    television_count,
    fridge_count,
    ac_count,
    average_ac_usage_hours
):
    daily_supply_hours = band_to_supply_hours(service_band)

    user_profile = {
        "household_size": household_size,
        "number_of_rooms": number_of_rooms,
        "daily_supply_hours": daily_supply_hours,
        "light_bulb_count": light_bulb_count,
        "fan_count": fan_count,
        "television_count": television_count,
        "fridge_count": fridge_count,
        "ac_count": ac_count,
        "average_ac_usage_hours": average_ac_usage_hours
    }

    return user_profile


def predict_monthly_consumption(user_profile):
    input_df = pd.DataFrame([user_profile])

    input_df = input_df[saved_items["model_features"]]

    predicted_kwh = saved_items["prediction_model"].predict(input_df)[0]

    predicted_kwh = max(predicted_kwh, 0)

    return predicted_kwh


def categorize_consumption(kwh):
    thresholds = saved_items["category_thresholds"]

    low_threshold = thresholds["low_threshold"]
    moderate_threshold = thresholds["moderate_threshold"]
    high_threshold = thresholds["high_threshold"]

    if kwh <= low_threshold:
        return "Low Consumption"
    elif kwh <= moderate_threshold:
        return "Moderate Consumption"
    elif kwh <= high_threshold:
        return "High Consumption"
    else:
        return "Very High Consumption"


def estimate_monthly_cost(predicted_kwh, tariff_per_kwh):
    return predicted_kwh * tariff_per_kwh


def assign_energy_segment(user_profile, predicted_kwh):
    cluster_input = user_profile.copy()
    cluster_input["estimated_monthly_kwh"] = predicted_kwh

    cluster_df = pd.DataFrame([cluster_input])
    cluster_df = cluster_df[saved_items["clustering_features"]]

    cluster_scaled = saved_items["cluster_scaler"].transform(cluster_df)

    cluster_number = int(saved_items["kmeans_model"].predict(cluster_scaled)[0])

    clean_mapping = {
        int(key): value for key, value in saved_items["cluster_segment_mapping"].items()
    }

    energy_segment = clean_mapping.get(
        cluster_number,
        "General Household Energy Segment"
    )

    return cluster_number, energy_segment


def detect_anomaly(user_profile, predicted_kwh):
    anomaly_input = user_profile.copy()
    anomaly_input["estimated_monthly_kwh"] = predicted_kwh

    anomaly_df = pd.DataFrame([anomaly_input])
    anomaly_df = anomaly_df[saved_items["anomaly_features"]]

    anomaly_scaled = saved_items["anomaly_scaler"].transform(anomaly_df)

    anomaly_label = int(saved_items["anomaly_model"].predict(anomaly_scaled)[0])
    anomaly_score = saved_items["anomaly_model"].decision_function(anomaly_scaled)[0]

    high_consumption_threshold = saved_items["anomaly_thresholds"]["high_consumption_threshold"]

    if anomaly_label == -1 and predicted_kwh >= high_consumption_threshold:
        anomaly_warning = "High Consumption Anomaly"
    elif anomaly_label == -1:
        anomaly_warning = "Unusual Low Consumption"
    else:
        anomaly_warning = "Normal Usage"

    return anomaly_warning, anomaly_score


def get_anomaly_message(anomaly_warning):
    if anomaly_warning == "High Consumption Anomaly":
        return "This household appears to have unusually high electricity consumption for its profile. Review high-energy appliances and check for possible waste."

    elif anomaly_warning == "Unusual Low Consumption":
        return "This household has unusually low estimated consumption. This may be due to limited supply or very few appliances."

    else:
        return "No unusual electricity usage pattern was detected."


def calculate_appliance_breakdown(user_profile):
    daily_supply_hours = user_profile["daily_supply_hours"]

    appliance_data = [
        {
            "Appliance": "Lighting",
            "Count": user_profile["light_bulb_count"],
            "Wattage": 10,
            "Daily Hours": min(daily_supply_hours, 6)
        },
        {
            "Appliance": "Fans",
            "Count": user_profile["fan_count"],
            "Wattage": 60,
            "Daily Hours": min(daily_supply_hours, 8)
        },
        {
            "Appliance": "Television",
            "Count": user_profile["television_count"],
            "Wattage": 100,
            "Daily Hours": min(daily_supply_hours, 4)
        },
        {
            "Appliance": "Fridge / Freezer",
            "Count": user_profile["fridge_count"],
            "Wattage": 150,
            "Daily Hours": 12 if user_profile["fridge_count"] > 0 else 0
        },
        {
            "Appliance": "Air Conditioner",
            "Count": user_profile["ac_count"],
            "Wattage": 1000,
            "Daily Hours": user_profile["average_ac_usage_hours"]
        }
    ]

    breakdown_df = pd.DataFrame(appliance_data)

    breakdown_df["Estimated Monthly kWh"] = (
        breakdown_df["Count"] *
        breakdown_df["Wattage"] *
        breakdown_df["Daily Hours"] *
        30 / 1000
    )

    breakdown_df = breakdown_df.sort_values(
        by="Estimated Monthly kWh",
        ascending=True
    )

    return breakdown_df


def generate_recommendations(
    user_profile,
    predicted_kwh,
    consumption_category,
    energy_segment,
    anomaly_warning,
    tariff_per_kwh,
    max_recommendations=3
):
    recommendations = []

    def add_recommendation(focus_area, message, monthly_kwh_saving=0):
        monthly_cost_saving = monthly_kwh_saving * tariff_per_kwh

        recommendations.append({
            "Focus Area": focus_area,
            "Recommendation": message,
            "Potential Monthly Saving": f"{round(monthly_kwh_saving, 2)} kWh / ₦{round(monthly_cost_saving, 2):,.2f}"
        })

    if anomaly_warning == "High Consumption Anomaly":
        add_recommendation(
            "Usage Check",
            "Your usage looks unusually high for this household profile. Review high-energy appliances and check for possible waste.",
            0
        )

    if user_profile["ac_count"] > 0 and user_profile["average_ac_usage_hours"] >= 3:
        saving_kwh = user_profile["ac_count"] * 1000 * 1 * 30 / 1000

        add_recommendation(
            "Air Conditioner",
            "Reduce AC usage by at least 1 hour daily or use fan support when possible.",
            saving_kwh
        )

    if user_profile["light_bulb_count"] >= 6:
        bulbs_to_reduce = min(user_profile["light_bulb_count"], 5)

        saving_kwh = bulbs_to_reduce * 10 * 2 * 30 / 1000

        add_recommendation(
            "Lighting",
            "Switch off unused bulbs and use LED bulbs where possible.",
            saving_kwh
        )

    if user_profile["fan_count"] >= 3:
        saving_kwh = user_profile["fan_count"] * 60 * 1 * 30 / 1000

        add_recommendation(
            "Fans",
            "Turn off fans in empty rooms and use natural ventilation when available.",
            saving_kwh
        )

    if user_profile["fridge_count"] > 0:
        add_recommendation(
            "Fridge / Freezer",
            "Avoid frequent opening, check door seals, and do not overload the fridge.",
            0
        )

    if consumption_category in ["High Consumption", "Very High Consumption"]:
        add_recommendation(
            "Energy Monitoring",
            "Track your high-use appliances weekly and reduce unnecessary usage during long supply hours.",
            0
        )

    if user_profile["daily_supply_hours"] <= 8:
        add_recommendation(
            "Limited Supply",
            "Prioritize essential appliances during limited supply hours to reduce waste.",
            0
        )

    if len(recommendations) == 0:
        add_recommendation(
            "Good Usage Pattern",
            "Your household energy profile looks reasonable. Continue switching off appliances when not in use.",
            0
        )

    recommendation_df = pd.DataFrame(recommendations)

    recommendation_df = recommendation_df.drop_duplicates(
        subset=["Focus Area"],
        keep="first"
    )

    recommendation_df = recommendation_df.head(max_recommendations)

    return recommendation_df


def get_consumption_gauge_value(consumption_category):
    gauge_map = {
        "Low Consumption": 25,
        "Moderate Consumption": 50,
        "High Consumption": 75,
        "Very High Consumption": 95
    }

    return gauge_map.get(consumption_category, 50)


# App Header

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">⚡ Household Energy AI</div>
        <div class="hero-subtitle">
            Predict monthly household electricity consumption, estimate cost, detect unusual usage,
            and receive simple energy-saving recommendations for Nigerian homes.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar Input Form

st.sidebar.title("🏠 Household Profile")
st.sidebar.write("Enter a simple household energy profile.")

household_size = st.sidebar.number_input(
    "Household size",
    min_value=1,
    max_value=30,
    value=4,
    step=1
)

number_of_rooms = st.sidebar.number_input(
    "Number of rooms",
    min_value=1,
    max_value=20,
    value=3,
    step=1
)

service_band = st.sidebar.selectbox(
    "Electricity service band",
    [
        "Band A",
        "Band B",
        "Band C",
        "Band D",
        "Band E",
        "Below Band E / Low Supply"
    ],
    index=0
)

daily_supply_hours = band_to_supply_hours(service_band)

st.sidebar.info(f"Estimated daily supply: {daily_supply_hours} hours")

tariff_per_kwh = st.sidebar.number_input(
    "Tariff per kWh (₦)",
    min_value=1.0,
    max_value=1000.0,
    value=225.0,
    step=5.0
)

st.sidebar.divider()

st.sidebar.subheader("Appliances")

light_bulb_count = st.sidebar.number_input(
    "Number of bulbs",
    min_value=0,
    max_value=100,
    value=8,
    step=1
)

fan_count = st.sidebar.number_input(
    "Number of fans",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

television_count = st.sidebar.number_input(
    "Number of TVs",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)

fridge_count = st.sidebar.number_input(
    "Number of fridges/freezers",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

ac_count = st.sidebar.number_input(
    "Number of ACs",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

average_ac_usage_hours = st.sidebar.slider(
    "Average AC usage hours per day",
    min_value=0,
    max_value=24,
    value=4,
    step=1
)

run_button = st.sidebar.button("Run Energy Prediction", use_container_width=True)


# Default Prediction State

if run_button:
    user_profile = create_user_profile(
        household_size=household_size,
        number_of_rooms=number_of_rooms,
        service_band=service_band,
        light_bulb_count=light_bulb_count,
        fan_count=fan_count,
        television_count=television_count,
        fridge_count=fridge_count,
        ac_count=ac_count,
        average_ac_usage_hours=average_ac_usage_hours
    )

    predicted_kwh = predict_monthly_consumption(user_profile)

    estimated_cost = estimate_monthly_cost(
        predicted_kwh,
        tariff_per_kwh
    )

    consumption_category = categorize_consumption(predicted_kwh)

    cluster_number, energy_segment = assign_energy_segment(
        user_profile,
        predicted_kwh
    )

    anomaly_warning, anomaly_score = detect_anomaly(
        user_profile,
        predicted_kwh
    )

    anomaly_message = get_anomaly_message(anomaly_warning)

    breakdown_df = calculate_appliance_breakdown(user_profile)

    recommendation_df = generate_recommendations(
        user_profile=user_profile,
        predicted_kwh=predicted_kwh,
        consumption_category=consumption_category,
        energy_segment=energy_segment,
        anomaly_warning=anomaly_warning,
        tariff_per_kwh=tariff_per_kwh,
        max_recommendations=3
    )


    # Main Result Cards
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted monthly use</div>
                <div class="metric-value">{predicted_kwh:,.2f} kWh</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated monthly cost</div>
                <div class="metric-value">₦{estimated_cost:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Consumption category</div>
                <div class="metric-value">{consumption_category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Energy segment</div>
                <div class="metric-value" style="font-size:20px;">{energy_segment}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


   # Consumption Gauge and Breakdown Chart
    
    left_col, right_col = st.columns([1, 1.4])

    with left_col:
        gauge_value = get_consumption_gauge_value(consumption_category)

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={"text": "Energy intensity level"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 60], "color": "#fef9c3"},
                        {"range": [60, 85], "color": "#fed7aa"},
                        {"range": [85, 100], "color": "#fecaca"}
                    ]
                }
            )
        )

        gauge_fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(gauge_fig, use_container_width=True)

    with right_col:
        breakdown_plot_df = breakdown_df[
            breakdown_df["Estimated Monthly kWh"] > 0
        ].copy()

        if len(breakdown_plot_df) > 0:
            bar_fig = px.bar(
                breakdown_plot_df,
                x="Estimated Monthly kWh",
                y="Appliance",
                orientation="h",
                text="Estimated Monthly kWh",
                title="Appliance Energy Breakdown"
            )

            bar_fig.update_traces(
                texttemplate="%{text:.1f} kWh",
                textposition="outside"
            )

            bar_fig.update_layout(
                height=360,
                xaxis_title="Estimated monthly kWh",
                yaxis_title="",
                margin=dict(l=20, r=40, t=60, b=20)
            )

            st.plotly_chart(bar_fig, use_container_width=True)

        else:
            st.info("No appliance energy breakdown available for this household profile.")


   # Anomaly Message
    
    if anomaly_warning == "Normal Usage":
        st.markdown(
            f"""
            <div class="safe-card">
                <b>✅ Usage Pattern:</b> {anomaly_message}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="warning-card">
                <b>⚠️ Anomaly Alert:</b> {anomaly_message}
            </div>
            """,
            unsafe_allow_html=True
        )


   # Recommendation Cards
    
    st.markdown(
        """
        <div class="section-card">
            <h3>💡 Top Energy-Saving Recommendations</h3>
            <p class="small-note">Showing only the top 3 recommendations to keep the dashboard clean.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    for index, row in recommendation_df.iterrows():
        st.markdown(
            f"""
            <div class="recommendation-card">
                <b>{row['Focus Area']}</b><br>
                {row['Recommendation']}<br>
                <span class="small-note">Potential monthly saving: {row['Potential Monthly Saving']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


   # Optional Details
   
    with st.expander("View detailed appliance breakdown table"):
        st.dataframe(
            breakdown_df.sort_values(
                by="Estimated Monthly kWh",
                ascending=False
            ),
            use_container_width=True
        )

    with st.expander("View model input profile"):
        input_preview = pd.DataFrame([user_profile])
        input_preview["tariff_per_kwh"] = tariff_per_kwh
        input_preview["service_band"] = service_band
        input_preview["cluster_number"] = cluster_number
        input_preview["anomaly_score"] = anomaly_score

        st.dataframe(input_preview, use_container_width=True)


else:
    st.markdown(
        """
        <div class="section-card">
            <h3>👈 Start from the sidebar</h3>
            <p>
                Enter a simple household profile and click <b>Run Energy Prediction</b>.
                The app will estimate monthly electricity use, cost, usage category,
                household energy segment, anomaly status, and top recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Footer Note

st.markdown(
    """
    <br>
    <p class="small-note">
    Note: This app estimates household electricity consumption using machine learning and appliance-based energy assumptions.
    </p>
    """,
    unsafe_allow_html=True
)
