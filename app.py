import streamlit as st
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from parser import parse_posturography_data
from analysis import (
    analyze_metric_data,
    analyze_radar_data,
    RADAR_KEY_METRICS_CONFIG,
    get_default_metric_index,
)

st.set_page_config(
    page_title="Posturography Analysis",
    page_icon=" BALANCE:",
    layout="wide",
)


def initialize_session_state():
    """Ensure session state keys exist before use."""
    if "page_init" not in st.session_state:
        st.session_state.all_parsed_data = []
        st.session_state.all_metrics = []
        st.session_state.all_test_conditions = []
        st.session_state.uploaded_file_names = []
        st.session_state.interpretation = (
            "Click the button to generate an AI-powered clinical interpretation..."
        )
        st.session_state.generating_ai = False
        st.session_state.current_analysis = None
        st.session_state.page_init = True


def reset_session():
    """Reset user selections without dropping the page init flag."""
    for key in list(st.session_state.keys()):
        if key != "page_init":
            del st.session_state[key]
    initialize_session_state()
    st.rerun()


initialize_session_state()

st.title(" 📈 Posturography Analysis Dashboard")
st.markdown("Upload, analyze, and interpret posturography data sets.")

with st.sidebar:
    st.header("1. Upload Data Files")
    uploaded_files = st.file_uploader(
        "Upload one or more .txt files", type="txt", accept_multiple_files=True
    )

    st.button("Reset and Clear All", on_click=reset_session, use_container_width=True)

    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]
        previous_file_names = st.session_state.get("uploaded_file_names", [])
        if previous_file_names != current_file_names:
            reset_session()
            st.session_state.uploaded_file_names = current_file_names

            all_parsed_data = []
            all_metrics = set()
            all_test_conditions = set()

            with st.spinner("Parsing files..."):
                for file in uploaded_files:
                    try:
                        content = file.getvalue().decode("utf-8")
                        parsed = parse_posturography_data(content, file.name)
                        if parsed:
                            all_parsed_data.append(parsed)
                            for test_key, test_data in parsed.get("tests", {}).items():
                                all_test_conditions.add(test_key)
                                for metric in test_data.get("metrics", {}).keys():
                                    all_metrics.add(metric)
                    except Exception as e:
                        st.error(f"Error parsing {file.name}: {e}")

            if all_parsed_data:
                all_parsed_data.sort(key=lambda x: x.get("date", datetime.now()))
                st.session_state.all_parsed_data = all_parsed_data
                st.session_state.all_metrics = sorted(list(all_metrics))
                st.session_state.all_test_conditions = sorted(list(all_test_conditions))
                st.success(f"Successfully parsed {len(all_parsed_data)} files.")
                st.rerun()

    if st.session_state.all_parsed_data:
        st.divider()
        st.header("2. Select Analysis")

        analysis_mode = st.radio(
            "Analysis Mode",
            ["Metric Comparison (Bar Chart)", "Comprehensive View (Radar Chart)"],
            key="analysis_mode",
        )

        analysis = None

        if analysis_mode == "Metric Comparison (Bar Chart)":
            default_index = get_default_metric_index(st.session_state.all_metrics)
            selected_metric = st.selectbox(
                "Select Metric",
                st.session_state.all_metrics,
                index=default_index,
                key="selected_metric",
            )
            if selected_metric:
                analysis = analyze_metric_data(
                    selected_metric,
                    st.session_state.all_parsed_data,
                )
        else:
            selected_test = st.selectbox(
                "Select Test Condition",
                st.session_state.all_test_conditions,
                key="selected_test",
            )

            st.markdown("**Select Radar Metrics**")
            selected_radar_metrics = []
            cols = st.columns(2)
            metrics_to_show = [
                m for m in RADAR_KEY_METRICS_CONFIG.keys() if m in st.session_state.all_metrics
            ]

            for i, metric in enumerate(metrics_to_show):
                with cols[i % 2]:
                    if st.checkbox(metric, value=True, key=f"radar_cb_{metric}"):
                        selected_radar_metrics.append(metric)

            if selected_test and selected_radar_metrics:
                analysis = analyze_radar_data(
                    selected_test,
                    selected_radar_metrics,
                    st.session_state.all_parsed_data,
                )

        if analysis:
            st.session_state.current_analysis = analysis

if not st.session_state.all_parsed_data:
    st.info("📊 Waiting for data... Upload one or more posturography .txt files to begin.")
    st.stop()

if st.session_state.current_analysis:
    analysis = st.session_state.current_analysis

    st.header("3. Visualization")
    if analysis.get("fig"):
        st.plotly_chart(analysis["fig"], use_container_width=True)
        if analysis["type"] == "radar":
            st.caption(
                "**Disclaimer:** Values are normalized to a 0-100 scale for visual comparison. "
                "'100' represents the 'best' score (e.g., high Stability Score, low Sway Path). "
                "Tooltips show original values."
            )
    else:
        st.warning("Could not generate chart. Please check your selections.")

    st.divider()

    st.header("Data Summary")
    if analysis.get("table_data") is not None:
        st.dataframe(analysis["table_data"], use_container_width=True)
    else:
        st.warning("Could not generate summary table.")

    st.divider()

    st.header("4. Clinical Interpretation")

    if st.button("Generate Interpretation", type="primary", disabled=st.session_state.generating_ai):
        st.session_state.generating_ai = True
        st.session_state.interpretation = "Generating... Please wait."
        st.rerun()

    if st.session_state.generating_ai:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.session_state.interpretation = (
                    "Error: GEMINI_API_KEY not found in st.secrets. Please add it."
                )
                st.session_state.generating_ai = False
                st.rerun()

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=analysis["ai_system_prompt"],
            )

            table_string = analysis["table_data"].to_string()
            user_query = analysis["ai_user_query"].replace("---SUMMARY_TEXT---", table_string)

            response = model.generate_content(user_query)
            st.session_state.interpretation = response.text

        except Exception as e:
            st.session_state.interpretation = f"An error occurred: {e}"
        finally:
            st.session_state.generating_ai = False
            st.rerun()

    st.markdown(st.session_state.interpretation, unsafe_allow_html=True)
