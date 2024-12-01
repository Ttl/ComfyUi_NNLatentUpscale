import torch
from .latent_resizer import LatentResizer
from comfy import model_management
import os


class NNLatentUpscale:
    """
    Upscales SDXL latent using neural network
    """

    def __init__(self):
        self.scale_factor = 0.13025
        self.dtype = torch.float32
        if model_management.should_use_fp16():
            self.dtype = torch.float16
        self.model_name = None

    @staticmethod
    def model_list():
        local_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(local_dir, "models")
        models = []
        for file in os.listdir(model_dir):
            if file.endswith(".pt"):
                models.append(file)
        return models

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "model": (NNLatentUpscale.model_list(),),
                "upscale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, latent, model, upscale):
        device = model_management.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        if model != self.model_name:
            self.model_name = model
            model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
            model_path = os.path.join(model_dir, model)
            self.model = LatentResizer.load_model(
                model_path, device, self.dtype
            )

        self.model.to(device=device)
        latent_out = (
            self.model(self.scale_factor * samples, scale=upscale) / self.scale_factor
        )

        if self.dtype != torch.float32:
            latent_out = latent_out.to(dtype=torch.float32)

        latent_out = latent_out.to(device="cpu")

        self.model.to(device=model_management.vae_offload_device())
        return ({"samples": latent_out},)
