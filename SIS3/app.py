import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
IMAGES_DIR = Path(__file__).parent / "images"

SPECIES_INFO = {
    "Adelie": (
        "Adélie penguin: a medium-sized penguin of the Antarctic coast. "
        "It has a black head, a white ring around each eye and a short beak "
        "that is mostly covered with feathers."
    ),
    "Chinstrap": (
        "Chinstrap penguin: named after the thin black band under its chin that "
        "looks like a helmet strap. It lives on Antarctic islands and is known "
        "for being bold and noisy."
    ),
    "Gentoo": (
        "Gentoo penguin: the largest of the three species, with a bright "
        "orange-red bill and a white patch above each eye. It is also the "
        "fastest swimmer among penguins, reaching about 36 km/h."
    ),
}


def find_image(species: str):
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = IMAGES_DIR / f"{species.lower()}.{ext}"
        if path.exists():
            return path
    return None


st.set_page_config(page_title="Penguin Classifier", page_icon="🐧", layout="wide")
st.title("🐧 Penguin species")
st.write("Enter measurements and gender.")

with st.sidebar:
    st.header("Status API")
    try:
        ok = requests.get(f"{API_URL}/", timeout=3).status_code == 200
    except requests.RequestException:
        ok = False
    st.success("API works") if ok else st.error("API failed")
    st.caption(f"Address: {API_URL}")

form_col, result_col = st.columns(2)

with form_col:
    with st.form("penguin_form"):
        bill_length = st.number_input("Bill length, mm", 20.0, 70.0, 39.1, 0.1)
        bill_depth = st.number_input("Bill depth, mm", 10.0, 30.0, 18.7, 0.1)
        flipper = st.number_input("Flipper length, mm", 150.0, 250.0, 181.0, 1.0)
        mass = st.number_input("Body mass, g", 2000.0, 7000.0, 3750.0, 50.0)
        sex_label = st.selectbox("Gender", ["Female", "Male"])
        submitted = st.form_submit_button("Predict")

with result_col:
    if submitted:
        payload = {
            "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth,
            "flipper_length_mm": flipper,
            "body_mass_g": mass,
            "sex": 0 if sex_label.startswith("Самка") else 1,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            species = result["prediction"]

            st.success(f"Predicted species: **{species}**")

            image_path = find_image(species)
            if image_path:
                st.image(str(image_path), caption=species, use_container_width=True)
            else:
                st.info(f"Image not found: add file images/{species.lower()}.jpg")

            if species in SPECIES_INFO:
                st.caption(SPECIES_INFO[species])

            st.subheader("Class probabilities")
            st.bar_chart(pd.Series(result["probabilities"]))
        except requests.ConnectionError:
            st.error("Connection error. Check API.")
        except requests.RequestException as exc:
            st.error(f"ERROR: {exc}")
    else:
        st.info("Result:")