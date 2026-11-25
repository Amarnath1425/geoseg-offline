import os
import sys
import numpy as np
from PIL import Image
from src.predict import Segmenter
from src.pipeline import bbox_from_center, process, overlay


def main():
    path = sys.argv[1]                                    # first thing after the command: the image path
    lat = float(sys.argv[2]) if len(sys.argv) > 2 else 19.3149    # optional: centre latitude
    lon = float(sys.argv[3]) if len(sys.argv) > 3 else 84.7941    # optional: centre longitude
    width_m = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0  # optional: width in metres

    img = Image.open(path).convert("RGB")                 # force 3 colour channels
    if max(img.size) > 1536:                              # shrink huge photos so it stays quick
        img.thumbnail((1536, 1536))
    rgb = np.array(img)                                   # picture -> grid of numbers
    h, w = rgb.shape[:2]
    bbox = bbox_from_center(lat, lon, width_m, width_m * h / w)   # keep the real aspect ratio

    seg = Segmenter()                                     # loads the ONNX or PyTorch model
    res = process(rgb, bbox, seg)                         # predict, then make polygons

    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]    # file name without ".png"
    Image.fromarray(overlay(rgb, res["mask"])).save(f"outputs/{base}_overlay.png")
    with open(f"outputs/{base}.geojson", "w") as f:
        f.write(res["geojson"])
    print(res["stats"], f"{res['seconds']:.1f}s via {seg.backend}")


if __name__ == "__main__":
    main()