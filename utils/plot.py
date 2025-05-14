import plotly.express as px
import streamlit as st
import pandas as pd

def smart_plot(df: pd.DataFrame):
    if df.empty:
        st.info("No data to visualize.")
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()

    chart_options = ["Bar", "Pie", "Scatter", "Line", "Table"]

    default_chart = "Table"
    default_x = None
    default_y = None

    if len(date_cols) == 1 and len(numeric_cols) >= 1:
        default_chart = "Line"
        default_x = date_cols[0]
        default_y = numeric_cols[0]
    elif len(cat_cols) == 1 and len(numeric_cols) >= 1:
        default_chart = "Bar"
        default_x = cat_cols[0]
        default_y = numeric_cols[0]


        if df[default_x].nunique() <= 8:
            default_chart = "Pie"
    elif len(numeric_cols) >= 2:
        default_chart = "Scatter"
        default_x = numeric_cols[0]
        default_y = numeric_cols[1]

    chart_col, x_col_sel, y_col_sel = st.columns([1, 2, 2])

    with chart_col:
        st.markdown("#### Chart Type")
        chart_type = st.selectbox(" ", chart_options, index=chart_options.index(default_chart), label_visibility="collapsed")

    with x_col_sel:
        st.markdown("#### X-axis")
        x_options = cat_cols + date_cols + numeric_cols
        x_default = default_x if default_x in x_options else (x_options[0] if x_options else None)
        x_col = st.selectbox(" ", x_options, index=x_options.index(x_default) if x_default else 0, label_visibility="collapsed") if x_options else None

    with y_col_sel:
        st.markdown("#### Y-axis")
        y_default = default_y if default_y in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        y_col = st.selectbox(" ", numeric_cols, index=numeric_cols.index(y_default) if y_default else 0, label_visibility="collapsed") if numeric_cols else None

    try:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col)
        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col)
        elif chart_type == "Pie":
            fig = px.pie(df, names=x_col, values=y_col)
        elif chart_type == "Table":
            st.dataframe(df)
            return

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to render chart: {e}")
