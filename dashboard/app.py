"""Entry point of the dashboard: page settings and navigation.

Run it from the project folder with `python -m streamlit run dashboard/app.py`.
Streamlit runs this file again at every interaction, then the page that is open.
"""

import streamlit as st

st.set_page_config(page_title="Turbofan RUL", page_icon=":material/flight:", layout="wide")

pages = [
    st.Page(
        "views/overview.py",
        title="Overview",
        icon=":material/dashboard:",
        default=True,
        help="Key figures, the pipeline and when each stage last wrote its outputs.",
    ),
    st.Page(
        "views/data_explorer.py",
        title="Data explorer",
        icon=":material/table_chart:",
        help="The raw NASA files: one engine's sensors, the fleet's lifetimes, correlations.",
    ),
    st.Page(
        "views/preprocessing.py",
        title="Preprocessing",
        icon=":material/tune:",
        help="Four before/after views: scaling, capped target, engine split, rolling window.",
    ),
    st.Page(
        "views/models.py",
        title="Models",
        icon=":material/model_training:",
        help="The six models compared on validation and test, errors, learning curves.",
    ),
]

st.navigation(pages).run()
