# GeoSeg Offline

**On-device semantic segmentation of map imagery: buildings and roads in, georeferenced GeoJSON out.**

---

## The problem

Web map services give us map pictures, not data. Turning those pictures into usable shapes (buildings, roads) is mostly done by hand, and most AI mapping tools need the cloud and an internet connection. Field teams, such as disaster responders or surveyors in low-connectivity areas, cannot rely on that.

## What GeoSeg Offline does

1. Takes a map area (a bounding box) and fetches the image from a Web Map Service (WMS), or loads an image you upload.
2. Runs a deep-learning segmentation model **on the local device** to label every pixel as *other*, *building* or *road*.
3. Converts the mask into polygons with real-world coordinates.
4. Exports a **GeoJSON** file that opens directly in QGIS, plus an overlay image and simple statistics (building regions, building area, road area, processing time).

The system is designed to run without the cloud: inference happens locally, and fetched map images are cached on disk so a repeated area needs no connection.

## How it works

```
Map area (bbox) -> WMS fetch (or upload) -> padding to model size
   -> U-Net (ResNet34 encoder) -> class mask
   -> vectorize (rasterio + shapely) -> GeoJSON + overlay + stats
```

| Module | File | Job |
|---|---|---|
| Data | `src/dataset.py` | Loads LoveDA image and mask pairs, random crops and flips, remaps 8 classes to 3 |
| Model | `src/model.py` | U-Net with a pretrained ResNet34 encoder (`segmentation-models-pytorch`) |
| Training | `src/train.py` | Class-weighted cross-entropy plus Dice loss; reports IoU per class on held-out images |
| Inference | `src/predict.py` | Runs the ONNX model if present, otherwise the PyTorch checkpoint |
| Export | `src/export_onnx.py` | Exports the trained model to ONNX and compares it against PyTorch |
| Pipeline | `src/pipeline.py` | WMS fetch with caching, georeferencing, polygons, statistics |
| Single image | `src/run_one.py` | Command-line run on one image |
| App | `app/app.py` | Streamlit interface |

## Data and training

- **Dataset:** [LoveDA](https://github.com/Junjue-Wang/LoveDA) (aerial imagery, 1024x1024). Mask classes used: 2 = building, 3 = road; everything else is merged into "other".
- **Training data:** the Urban scenes of the LoveDA Val split (677 images). The last 70 images are held out and never used for training, so evaluation is on images the model has not seen. This smaller setup was chosen because of the short hackathon time frame.
- **Setup:** 256x256 random crops, flips, Adam (learning rate 3e-4), and a 3x loss weight on building and road because they cover few pixels.
- **Metric:** intersection over union (IoU) per class, printed after every epoch by `src/train.py`. The best checkpoint is saved to `models/best.pth`.

## Setup

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/Amarnath1425/geoseg-offline.git
cd geoseg-offline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets and model weights are not stored in the repo. To train, download LoveDA so that `data/Val/Urban/images_png` and `data/Val/Urban/masks_png` exist. The first training run downloads the ImageNet ResNet34 weights once, which needs the internet.

## Usage

Run all commands from the project root with the virtual environment active.

```bash
# train (the number is the epoch count)
python -m src.train 8

# optional: export to ONNX
python -m src.export_onnx

# one image from the command line (optional: latitude, longitude, width in metres)
python -m src.run_one path/to/image.png 19.3149 84.7941 300

# the app
streamlit run app/app.py
```

`run_one` writes an overlay PNG and a GeoJSON to `outputs/`. In the app, choose **Upload an image**, enter the centre latitude, longitude and width of the area, and click **Run**.

> An uploaded image carries no location. The polygons are placed using the coordinates and width you enter, so wrong inputs shift the shapes in QGIS.

## Limitations

- **Roads are the harder class.** They are thin and often broken by trees or shadows, and the human-drawn training labels are coarse, so road accuracy is expected to be lower than building accuracy.
- **Polygons, not individual buildings.** Touching buildings merge into one polygon, so "buildings found" counts building regions, not exact houses.
- **Roads are areas, not networks.** The output is road shapes, not centrelines or routes.
- **Domain gap.** The model is trained on LoveDA-style aerial photos (about 0.3 m per pixel). Blurry imagery, very different zoom levels, or drawn street maps will give poor results.
- **Small training set** because of the time limit. More data and longer training would improve results.
- A new area fetched from a WMS needs internet once; the image is then cached.

## Future scope

More classes (water, vegetation), mobile deployment (TFLite), more map sources, fine-tuning on local imagery, and road centreline extraction.

## References

- LoveDA dataset: Wang et al., *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation*
- U-Net: Ronneberger, Fischer, Brox (2015), arXiv:1505.04597
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch), PyTorch, ONNX Runtime, rasterio, Shapely, GeoPandas, Streamlit, QGIS

## License

See the `LICENSE` file.