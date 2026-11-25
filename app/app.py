import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))   # so "src" can be imported

import numpy as np
import streamlit as st
from PIL import Image
from src.predict import Segmenter
from src.pipeline import fetch_wms, bbox_from_center, process, overlay

st.set_page_config(page_title="GeoSeg Offline", layout="wide")
st.title("GeoSeg Offline: buildings and roads from map imagery")


@st.cache_resource                                    # load the model once, not on every click
def get_segmenter():
    return Segmenter()


seg = get_segmenter()
st.caption(f"Model runs on this device with {seg.backend}. No cloud needed at runtime.")

mode = st.sidebar.radio("Image source", ["Upload an image", "WMS (cached after first fetch)"])
lat = st.sidebar.number_input("Centre latitude", value=19.3149, format="%.6f")
lon = st.sidebar.number_input("Centre longitude", value=84.7941, format="%.6f")
width_m = st.sidebar.number_input("Width of area in metres", value=300.0)
if mode.startswith("Upload"):
    file = st.sidebar.file_uploader("Aerial image", type=["png", "jpg", "jpeg"])
else:
    url = st.sidebar.text_input("WMS base URL")
    layer = st.sidebar.text_input("Layer name")
    version = st.sidebar.selectbox("WMS version", ["1.1.1", "1.3.0"])

if st.button("Run"):
    try:
        if mode.startswith("Upload"):
            if file is None:
                st.warning("Upload an image first.")
                st.stop()
            img = Image.open(file).convert("RGB")
            if max(img.size) > 1536:                  # keep it quick
                img.thumbnail((1536, 1536))
            rgb = np.array(img)
            h, w = rgb.shape[:2]
            bbox = bbox_from_center(lat, lon, width_m, width_m * h / w)
        else:
            bbox = bbox_from_center(lat, lon, width_m)
            rgb = fetch_wms(url, layer, bbox, version=version)
        st.session_state["res"] = (rgb, process(rgb, bbox, seg))
    except Exception as e:
        st.error(f"Failed: {e}")

if "res" in st.session_state:                         # stays on screen after the download click
    rgb, res = st.session_state["res"]
    c1, c2 = st.columns(2)
    c1.image(rgb, caption="Input map image")
    c2.image(overlay(rgb, res["mask"]), caption="Red = buildings, yellow = roads")
    s = res["stats"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Buildings found", s["buildings"])
    m2.metric("Building area (m²)", f"{s['building_area_m2']:.0f}")
    m3.metric("Road area (m²)", f"{s['road_area_m2']:.0f}")
    m4.metric("Processing time (s)", f"{res['seconds']:.1f}")
    st.download_button("Download GeoJSON", data=res["geojson"],
                       file_name="features.geojson", mime="application/geo+json")