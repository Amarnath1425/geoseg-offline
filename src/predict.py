import os
import numpy as np
import torch
from src.dataset import normalize
from src.model import build_model, get_device


class Segmenter:
    """Uses the ONNX model if it exists, otherwise the PyTorch checkpoint. Both run offline."""
    def __init__(self, onnx_path="models/model.onnx", pth_path="models/best.pth"):
        self.session, self.model = None, None
        if os.path.exists(onnx_path):
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.backend = "ONNX Runtime"
        else:
            self.device = get_device()
            self.model = build_model(pretrained=False)
            self.model.load_state_dict(torch.load(pth_path, map_location="cpu"))
            self.model.to(self.device).eval()
            self.backend = "PyTorch"

    def predict(self, rgb):
        """rgb: uint8 HxWx3 array -> uint8 HxW mask (0 other, 1 building, 2 road)."""
        h, w = rgb.shape[:2]
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32   # U-Net needs sizes divisible by 32
        padded = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        x = normalize(padded).unsqueeze(0)                # add the batch dimension
        if self.session is not None:
            out = self.session.run(None, {"input": x.numpy()})[0]
            pred = out[0].argmax(axis=0)
        else:
            with torch.no_grad():
                out = self.model(x.to(self.device))
            pred = out[0].argmax(dim=0).cpu().numpy()
        return pred[:h, :w].astype(np.uint8)              # cut the padding off again