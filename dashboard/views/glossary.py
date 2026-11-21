"""Glossary: the terms of the project, explained simply."""

from pathlib import Path

import streamlit as st
import yaml

st.title("Glossary")
st.caption(
    "The words used in this dashboard and in the project, explained in one or two sentences. "
    "Type in the box to filter."
)

sections = yaml.safe_load(
    (Path(__file__).parent.parent / "glossary.yaml").read_text(encoding="utf-8")
)
search = st.text_input("Search", placeholder="for example: leakage, SHAP, epoch").strip().lower()

shown = 0
for section, terms in sections.items():
    matching = [
        item
        for item in terms
        if search in item["term"].lower() or search in item["meaning"].lower()
    ]
    if matching:
        st.subheader(section)
        for item in matching:
            st.markdown(f"**{item['term']}**: {item['meaning']}")
        shown += len(matching)
if shown == 0:
    st.info(f"No term matches “{search}”.", icon=":material/search_off:")
