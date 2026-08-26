import os
import requests
import streamlit as st
import mlflow
from dotenv import load_dotenv

from prometheus_client.parser import text_string_to_metric_families

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")

st.set_page_config(page_title="Admin Dashboard", page_icon="\U0001F6E0", layout="wide")
st.header("Admin Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "MLflow Runs", "Trigger Training", "API Traffic"])

def get_selected_prometheus_metrics(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url}/metrics", timeout=10)
    response.raise_for_status()

    wanted = {"http_request_duration_seconds_count", "http_request_duration_seconds_sum"}
    result = {}

    for family in text_string_to_metric_families(response.text):
        if family.name in wanted:
            result[family.name] = [
                {"labels": sample.labels, "value": sample.value}
                for sample in family.samples
            ]
    return result

# Tab 1
with tab1:
    st.subheader("Current Registered Champions")
    try:
        response = requests.get(f"{API_BASE_URL}/model_metrics", timeout=15)
        if response.status_code == 200:
            data = response.json()["data"]
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Classification (EMI Eligibility)**")
                clf = data.get("classification")
                if clf:
                    st.metric("F1 Macro", f"{clf['metrics'].get('f1_macro', 0):.4f}")
                    st.metric("Accuracy", f"{clf['metrics'].get('accuracy', 0):.4f}")
                    st.metric("Critical Error Rate", f"{clf['metrics'].get('critical_error_rate', 0):.4f}")
                    st.caption(f"Version {clf['version']} | Run ID: {clf['run_id']}")
                else:
                    st.warning("No classification champion found yet.")

            with col2:
                st.markdown("**Regression (Max Monthly EMI)**")
                reg = data.get("regression")
                if reg:
                    st.metric("RMSE", f"{reg['metrics'].get('rmse', 0):.2f}")
                    st.metric("MAE", f"{reg['metrics'].get('mae', 0):.2f}")
                    st.metric("R²", f"{reg['metrics'].get('r2', 0):.4f}")
                    st.caption(f"Version {reg['version']} | Run ID: {reg['run_id']}")
                else:
                    st.warning("No regression champion found yet.")
        else:
            st.error("Could not fetch model metrics from API.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Tab 2
with tab2:
    st.subheader("Experiment Run History")
    exp_choice = st.selectbox("Experiment", ["classification experiment", "regression experiment"])
    try:
        response = requests.get(f"{API_BASE_URL}/experiment_runs", params={"experiment_name": exp_choice}, timeout=15)
        if response.status_code == 200:
            runs = response.json()["runs"]
            if runs:
                st.dataframe(runs)
            else:
                st.info("No runs logged yet for this experiment.")
        else:
            st.error("Could not fetch experiment runs from API.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# --- Tab 3: Trigger training pipeline ---
with tab3:
    st.subheader("Trigger Retraining")
    st.warning("This runs the full training pipeline (ingestion → transformation → training → evaluation → pushing). May take a while.")

    if st.button("Start Training Pipeline"):
        try:
            response = requests.post(f"{API_BASE_URL}/train", timeout=10)
            if response.status_code == 200:
                st.success("Training pipeline started in the background. Check API logs for progress.")
            else:
                st.error(f"Failed to start training: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not reach the API at {API_BASE_URL}. Is it running?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    st.divider()
    st.subheader("Reload Prediction Resources")
    st.caption("Call this after training completes to make the API use the new champion model.")
    if st.button("Reload Resources"):
        try:
            response = requests.post(f"{API_BASE_URL}/load_resources", timeout=30)
            if response.status_code == 200:
                st.success("Resources reloaded successfully.")
            else:
                st.error(f"Failed to reload: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


with tab4:
    st.subheader("API Traffic (Prometheus)")
    try:
        metrics_data = get_selected_prometheus_metrics(API_BASE_URL)

        if "http_request_duration_seconds_count" in metrics_data and "http_request_duration_seconds_sum" in metrics_data:
            count_samples = {s["labels"].get("handler"): s["value"] for s in metrics_data["http_request_duration_seconds_count"]}
            sum_samples = {s["labels"].get("handler"): s["value"] for s in metrics_data["http_request_duration_seconds_sum"]}

            st.markdown("**Requests & Average Latency by Endpoint**")
            rows = []
            for handler, count in count_samples.items():
                total_time = sum_samples.get(handler, 0)
                avg = total_time / count if count > 0 else 0
                rows.append({"handler": handler, "request_count": int(count), "avg_latency_sec": round(avg, 4)})
            st.dataframe(rows)
        else:
            st.info("No traffic data yet.")

    except Exception as e:
        st.error(f"Could not fetch Prometheus metrics: {e}")