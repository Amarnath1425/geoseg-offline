import numpy as np
import torch
from src.model import build_model


def main():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load("models/best.pth", map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 256, 256)               # example input that fixes the layout
    kwargs = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},   # any image size works
                      "logits": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, "models/model.onnx", dynamo=False, **kwargs)
    except TypeError:                                  # older torch has no dynamo option
        torch.onnx.export(model, dummy, "models/model.onnx", **kwargs)

    import onnxruntime as ort                          # check ONNX gives the same answers
    sess = ort.InferenceSession("models/model.onnx", providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 512, 512)                    # a different size on purpose
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    print("max difference vs PyTorch:", float(np.abs(ref - out).max()))   # should be tiny


if __name__ == "__main__":
    main()