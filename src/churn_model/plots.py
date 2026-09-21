"""Diagnostic plots. Maps to notebook cells 14, 17, 20, 22 (visualization half).

Rebuilt in Plotly rather than matplotlib/seaborn: interactive by default,
and these figures get reused as-is in the Streamlit app later, where
Plotly renders natively (st.plotly_chart) - matplotlib figures would need
re-plotting for that surface.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Clean axes, no gridline clutter, journal-style whitespace.
PLOT_TEMPLATE = "simple_white"
FONT = dict(family="Arial", size=13)


def confusion_matrix_figure(cm, labels: list, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(title=title, template=PLOT_TEMPLATE, width=480, height=430, font=FONT)
    fig.update_yaxes(autorange="reversed")
    return fig


def feature_importance_figure(
    importance_df: pd.DataFrame, top_n: int = 10, title: str = "Feature Importance"
) -> go.Figure:
    top = importance_df.head(top_n).sort_values("Importance")
    fig = px.bar(
        top, x="Importance", y="Feature", orientation="h", title=title, template=PLOT_TEMPLATE
    )
    fig.update_layout(font=FONT, width=650, height=430)
    return fig


def coefficient_figure(coef_df: pd.DataFrame, title: str = "Logistic Regression Coefficients") -> go.Figure:
    ordered = coef_df.sort_values("Coefficient")
    fig = px.bar(
        ordered,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title=title,
        template=PLOT_TEMPLATE,
        color="Coefficient",
        color_continuous_scale="RdBu",
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(font=FONT, width=700, height=550, coloraxis_showscale=False)
    return fig