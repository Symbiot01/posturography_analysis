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

def analyze_radar_data(test, selected_radar_metrics, all_parsed_data, normalize=True):
    """Build radar view (normalized or raw) and supporting stats table."""
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

    metric_ranges = {}
    if normalize:
        for metric in selected_radar_metrics:
            values = []
            for data in all_parsed_data:
                val = (
                    data.get("tests", {})
                    .get(test, {})
                    .get("metrics", {})
                    .get(metric)
                )
                if val is not None:
                    values.append(val)

            if values:
                min_val = min(values)
                max_val = max(values)
            else:
                min_val = 0
                max_val = 0

            metric_ranges[metric] = (min_val, max_val)

    table_rows = []
    global_min_val = float("inf")
    global_max_val = float("-inf")

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
            original_data.append(float(val) if val is not None else np.nan)

            if val is None:
                normalized_data.append(np.nan)
                continue

            if normalize:
                min_val, max_val = metric_ranges[metric]
                orientation = RADAR_KEY_METRICS_CONFIG.get(metric, "high")

                if max_val == min_val:
                    norm_val = 1.0
                else:
                    norm_val = (val - min_val) / (max_val - min_val)
                    if orientation == "low":
                        norm_val = 1 - norm_val

                norm_val = float(np.clip(norm_val, 0, 1))
                normalized_data.append(norm_val * 100)
            else:
                normalized_data.append(val)
                global_min_val = min(global_min_val, val)
                global_max_val = max(global_max_val, val)

        hovertemplate = "<b>%{theta}</b><br>"
        if normalize:
            hovertemplate += (
                "Original: %{customdata:.3f}<br>Normalized: %{r:.1f}<extra></extra>"
            )
        else:
            hovertemplate += "Value: %{r:.3f}<extra></extra>"

        fig.add_trace(
            go.Scatterpolar(
                r=normalized_data,
                theta=selected_radar_metrics,
                fill="toself",
                name=file_names[i],
                customdata=original_data,
                hovertemplate=hovertemplate,
            )
        )

    if normalize:
        radialaxis = dict(visible=True, range=[0, 100], showticklabels=False)
        title_suffix = "Normalized"
    else:
        if global_min_val == float("inf") or global_max_val == float("-inf"):
            global_min_val, global_max_val = 0, 1
        elif global_min_val == global_max_val:
            global_max_val = global_min_val + 1
        elif global_min_val > 0:
            global_min_val = 0

        radialaxis = dict(
            visible=True,
            range=[global_min_val, global_max_val],
            showticklabels=True,
            tickformat=".2f",
        )
        title_suffix = "Original Values"

    fig.update_layout(
        title=f"Comprehensive Profile for {test} ({title_suffix})",
        polar=dict(
            radialaxis=radialaxis,
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
        "normalized": normalize,
    }

