import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the FIRST st command)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📊 Data Analyst Helper",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – light polish
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card { background: #f0f2f6; border-radius: 10px; padding: 12px; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data          # <-- Streamlit caching: avoids re-reading on every interaction
def load_data(uploaded_file):
    """Read CSV or Excel file into a DataFrame."""
    name = uploaded_file.name
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None


def get_df_info(df: pd.DataFrame) -> str:
    """Capture df.info() output as a string."""
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Data Analyst Helper")
    st.markdown("---")

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xls", "xlsx"],
        help="Supports CSV and Excel files",
    )

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate to",
        ["🏠 Overview", "📋 Data Preview", "📈 Visualizations", "🔍 Filter & Search", "📥 Export"],
    )

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# ─────────────────────────────────────────────
# LOAD DATA OR SHOW DEMO PROMPT
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.title("Welcome to the Data Analyst Helper 👋")
    st.info("👈  Upload a CSV or Excel file from the sidebar to get started.")

    # Show a sample demo using built-in dataset
    st.subheader("Or try with a sample dataset")
    if st.button("Load Titanic Sample Dataset 🚢"):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df_demo = pd.read_csv(url)
        st.session_state["demo_df"] = df_demo
        st.success("Sample dataset loaded! Navigate using the sidebar.")

    if "demo_df" in st.session_state:
        df = st.session_state["demo_df"]
    else:
        st.stop()  # <-- Stop rendering until data is available
else:
    df = load_data(uploaded_file)
    if df is None:
        st.stop()

# ─────────────────────────────────────────────
# PAGE 1 – OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏠 Dataset Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown("---")

    # Column type breakdown
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🗂 Column Types")
        type_counts = df.dtypes.astype(str).value_counts().reset_index()
        type_counts.columns = ["Data Type", "Count"]
        st.dataframe(type_counts, use_container_width=True)

    with col_b:
        st.subheader("❓ Missing Values per Column")
        missing = df.isnull().sum()
        missing = missing[missing > 0].reset_index()
        missing.columns = ["Column", "Missing Count"]
        if missing.empty:
            st.success("No missing values found!")
        else:
            missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
            st.dataframe(missing, use_container_width=True)

    st.markdown("---")
    st.subheader("ℹ️ DataFrame Info")
    with st.expander("Click to expand"):
        st.text(get_df_info(df))


# ─────────────────────────────────────────────
# PAGE 2 – DATA PREVIEW
# ─────────────────────────────────────────────
elif page == "📋 Data Preview":
    st.title("📋 Data Preview")

    # Row slider
    n_rows = st.slider("Number of rows to display", min_value=5, max_value=min(500, len(df)), value=20, step=5)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Descriptive Statistics")

    # Toggle between numeric and categorical stats
    stat_type = st.radio("Show statistics for:", ["Numeric columns", "Categorical columns"], horizontal=True)

    if stat_type == "Numeric columns":
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)
    else:
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.dataframe(df[cat_cols].describe().T, use_container_width=True)
        else:
            st.info("No categorical columns found.")


# ─────────────────────────────────────────────
# PAGE 3 – VISUALIZATIONS
# ─────────────────────────────────────────────
elif page == "📈 Visualizations":
    st.title("📈 Visualizations")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── Histogram ──────────────────────────────
    st.subheader("📊 Histogram")
    hist_col = st.selectbox("Select a numeric column", numeric_cols, key="hist")
    bins = st.slider("Number of bins", 5, 100, 20)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[hist_col].dropna(), bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(hist_col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {hist_col}")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Scatter Plot ───────────────────────────
    st.subheader("🔵 Scatter Plot")
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X axis", numeric_cols, key="scatter_x")
    y_col = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
    hue_col = c3.selectbox("Color by (optional)", ["None"] + cat_cols, key="scatter_hue")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if hue_col != "None":
        for category, group in df.groupby(hue_col):
            ax2.scatter(group[x_col], group[y_col], label=str(category), alpha=0.7, s=30)
        ax2.legend(title=hue_col, bbox_to_anchor=(1, 1))
    else:
        ax2.scatter(df[x_col], df[y_col], alpha=0.6, s=30, color="#4C72B0")
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.set_title(f"{x_col} vs {y_col}")
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # ── Correlation Heatmap ────────────────────
    st.subheader("🔥 Correlation Heatmap")
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for a heatmap.")
    else:
        selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:min(8, len(numeric_cols))])
        if len(selected_cols) >= 2:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            corr = df[selected_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3, linewidths=0.5)
            ax3.set_title("Correlation Matrix")
            st.pyplot(fig3)
            plt.close()

    st.markdown("---")

    # ── Bar Chart (Categorical) ────────────────
    if cat_cols:
        st.subheader("📊 Bar Chart (Categorical Column)")
        bar_col = st.selectbox("Select a categorical column", cat_cols)
        top_n = st.slider("Show top N categories", 3, 30, 10)

        counts = df[bar_col].value_counts().head(top_n)
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax4, color="#4C72B0", edgecolor="white")
        ax4.set_xlabel(bar_col)
        ax4.set_ylabel("Count")
        ax4.set_title(f"Top {top_n} values in '{bar_col}'")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig4)
        plt.close()


# ─────────────────────────────────────────────
# PAGE 4 – FILTER & SEARCH
# ─────────────────────────────────────────────
elif page == "🔍 Filter & Search":
    st.title("🔍 Filter & Search")

    filtered_df = df.copy()

    # ── Numeric filters ────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    with st.expander("🔢 Numeric Filters", expanded=True):
        for col in numeric_cols[:5]:  # Limit to first 5 to avoid clutter
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            if col_min < col_max:
                selected = st.slider(
                    f"{col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key=f"num_{col}",
                )
                filtered_df = filtered_df[(filtered_df[col] >= selected[0]) & (filtered_df[col] <= selected[1])]

    # ── Categorical filters ────────────────────
    with st.expander("🔤 Categorical Filters", expanded=True):
        for col in cat_cols[:3]:
            unique_vals = df[col].dropna().unique().tolist()
            selected_vals = st.multiselect(f"{col}", unique_vals, default=unique_vals, key=f"cat_{col}")
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    # ── Text Search ────────────────────────────
    with st.expander("🔍 Text Search", expanded=False):
        search_col = st.selectbox("Column to search in", df.columns.tolist())
        search_term = st.text_input("Search term")
        if search_term:
            filtered_df = filtered_df[
                filtered_df[search_col].astype(str).str.contains(search_term, case=False, na=False)
            ]

    st.markdown(f"**{len(filtered_df):,} rows** match your filters (out of {len(df):,})")
    st.dataframe(filtered_df, use_container_width=True)

    # Save filtered data in session state for export
    st.session_state["filtered_df"] = filtered_df


# ─────────────────────────────────────────────
# PAGE 5 – EXPORT
# ─────────────────────────────────────────────
elif page == "📥 Export":
    st.title("📥 Export Data")

    export_df = st.session_state.get("filtered_df", df)
    st.info(f"Exporting **{len(export_df):,} rows** and **{export_df.shape[1]} columns**.")
    st.caption("Tip: Go to the Filter & Search page first to export a filtered subset.")

    # CSV download
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_data,
        file_name="export.csv",
        mime="text/csv",
    )

    # Excel download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Data")
    st.download_button(
        label="⬇️ Download as Excel",
        data=buffer.getvalue(),
        file_name="export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )