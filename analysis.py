import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

RADAR_KEY_METRICS_CONFIG = {
    "Stability Score": "high",
    "Sway Path Length": "low",
    "Sway Velocity (Ave)": "low",
    "Area 95% Conf. Ellipse": "low",
    "Fatigue Ratio": "low",
    "Adaptation Ratio": "high",
    "Directionality": "low",
}

def get_default_metric_index(all_metrics):
    """Return Stability Score index when available to mimic prior UX."""
    if "Stability Score" in all_metrics:
        return all_metrics.index("Stability Score")
    return 0

def _calculate_stats(df, file_names):
    """Add change columns for n=2 or descriptive stats for n>2."""
    num_files = len(file_names)
    df_numeric = df[file_names].apply(pd.to_numeric, errors="coerce")

    if num_files == 2:
        df["Change"] = df_numeric[file_names[1]] - df_numeric[file_names[0]]
        df["% Change"] = (
            df["Change"] / df_numeric[file_names[0]].replace(0, np.nan)
        ) * 100
        df["% Change"] = df["% Change"].round(1)
        df["Change"] = df["Change"].round(3)
    elif num_files > 2:
        df["Mean"] = df_numeric.mean(axis=1).round(3)
        df["Std Dev"] = df_numeric.std(axis=1).round(3)

    return df.round(3)

def analyze_metric_data(metric, all_parsed_data):
    """Build grouped bar chart and summary for the selected metric."""
    file_names = [d["fileName"] for d in all_parsed_data]

    common_tests = set()
    for data in all_parsed_data:
        for test_key, test_data in data.get("tests", {}).items():
            if metric in test_data.get("metrics", {}):
                common_tests.add(test_key)
    sorted_tests = sorted(list(common_tests))

    data_to_plot = []
    for test in sorted_tests:
        for i, file_data in enumerate(all_parsed_data):
            val = (
                file_data.get("tests", {})
                .get(test, {})
                .get("metrics", {})
                .get(metric)
            )
            data_to_plot.append(
                {
                    "Test Condition": test,
                    "File": file_names[i],
                    "Value": val if val is not None else np.nan,
                }
            )

    df_plot = pd.DataFrame(data_to_plot).dropna()

    fig = px.bar(
        df_plot,
        x="Test Condition",
        y="Value",
        color="File",
        barmode="group",
        title=f"{metric} Comparison (n={len(file_names)} files)",
        labels={"Value": f"{metric} Value", "File": "Data File"},
    )
    fig.update_layout(legend_title_text="Data File")

    df_table = (
        df_plot.pivot(index="Test Condition", columns="File", values="Value")
        .reindex(sorted_tests)
        .reindex(columns=file_names)
    )

    df_table = _calculate_stats(df_table, file_names)
    df_table = df_table.reset_index().rename_axis(None, axis=1)
    df_table.insert(0, "Metric", metric)

    system_prompt = f"""
You are a clinical expert in vestibular and balance disorders, specializing in interpreting posturography data.
- Analyze the provided data summary for the SINGLE METRIC: {metric}.
- Provide a brief, clinically-oriented interpretation.
- If n=2, focus on the magnitude and direction of change.
- If n>2, focus on the mean and standard deviation.
- If n=1, describe the single-point data relative to typical expectations (e.g., high/low sway).
- Always mention which test conditions (e.g., "NSEO", "PSEC Rt") show the most significant findings.
- Be concise, professional, and use clear language. Do not make a diagnosis.
- Structure your response with a summary, key findings, and potential implications.
- Start with a clear heading: "### Interpretation for {metric}".
"""

    user_query = f"""
Here is a summary of posturography data for the metric "{metric}". Please provide a clinical interpretation based on this table.
---
---SUMMARY_TEXT---
---
"""

    return {
        "type": "metric",
        "fig": fig,
        "table_data": df_table,
        "ai_system_prompt": system_prompt,
        "ai_user_query": user_query,
    }

def analyze_radar_data(test, selected_radar_metrics, all_parsed_data):
    """Build normalized radar view and supporting stats table."""
    if not selected_radar_metrics:
        return {
            "type": "radar",
            "fig": None,
            "table_data": pd.DataFrame(),
            "ai_system_prompt": "",
            "ai_user_query": "",
        }

    file_names = [d["fileName"] for d in all_parsed_data]
    fig = go.Figure()

    max_values = {}
    for metric in selected_radar_metrics:
        max_val = 0
        for data in all_parsed_data:
            val = (
                data.get("tests", {})
                .get(test, {})
                .get("metrics", {})
                .get(metric)
            )
            if val is not None and abs(val) > max_val:
                max_val = abs(val)
        max_values[metric] = max_val if max_val > 0 else 1

    table_rows = []
    for i, file_data in enumerate(all_parsed_data):
        normalized_data = []
        original_data = []

        for metric in selected_radar_metrics:
            val = (
                file_data.get("tests", {})
                .get(test, {})
                .get("metrics", {})
                .get(metric)
            )
            original_data.append(val)

            if val is None:
                normalized_data.append(np.nan)
                continue

            norm_val = val / max_values[metric]

            if RADAR_KEY_METRICS_CONFIG.get(metric) == "low":
                norm_val = 1 - abs(norm_val)

            normalized_data.append(max(0, norm_val) * 100)

        fig.add_trace(
            go.Scatterpolar(
                r=normalized_data,
                theta=selected_radar_metrics,
                fill="toself",
                name=file_names[i],
                customdata=original_data,
                hovertemplate=(
                    "<b>%{theta}</b><br>Original: %{customdata:.3f}<br>"
                    "Normalized: %{r:.1f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Comprehensive Profile for {test} (Normalized)",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        legend_title_text="Data File",
    )

    for metric in selected_radar_metrics:
        row = {"Metric": metric}
        for i, file_name in enumerate(file_names):
            val = (
                all_parsed_data[i]
                .get("tests", {})
                .get(test, {})
                .get("metrics", {})
                .get(metric)
            )
            row[file_name] = val
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows).set_index("Metric")
    df_table = _calculate_stats(df_table, file_names)
    df_table = df_table.reset_index().rename_axis(None, axis=1)

    system_prompt = f"""
You are a clinical expert in vestibular and balance disorders, specializing in interpreting posturography data.
- Analyze the provided data summary, which shows a COMPREHENSIVE PROFILE for the SINGLE TEST CONDITION: {test}.
- The table shows ORIGINAL values for the selected key metrics.
- Provide a holistic interpretation of the patient's performance under this specific test condition ({test}), comparing the different files based only on the metrics provided in the table.
- For n=2, compare the overall profile change.
- For n>2, comment on the average profile and variability.
- For n=1, comment on the single profile.
- Be concise, professional, and use clear language. Do not make a diagnosis.
- Structure your response with a summary, key findings, and potential implications for this test condition.
- Start with a clear heading: "### Comprehensive Interpretation for {test}".
"""

    user_query = f"""
Here is a summary of key posturography metrics for the test condition "{test}". Please provide a holistic clinical interpretation comparing the files based only on these metrics.
---
---SUMMARY_TEXT---
---
"""

    return {
        "type": "radar",
        "fig": fig,
        "table_data": df_table,
        "ai_system_prompt": system_prompt,
        "ai_user_query": user_query,
    }
