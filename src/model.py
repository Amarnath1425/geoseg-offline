import torch
import segmentation_models_pytorch as smp


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # Apple GPU
        return "mps"
    return "cpu"


def build_model(pretrained=True):
    # U-Net: an encoder shrinks the image to find "what", a decoder grows it back to say "where".
    # resnet34 is the encoder; ImageNet weights give it a head start.
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=3,          # other, building, road
    )