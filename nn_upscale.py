import os

import torch
from comfy import model_management

from .latent_resizer import LatentResizer


class NNLatentUpscale:
    """
    Upscales SDXL/Flux latent using neural network
    """

    def __init__(self):
        self.local_dir = os.path.dirname(os.path.realpath(__file__))
        self.scale_factor = {"SDXL": 0.13025, "SD 1.x": 0.13025, "Flux1.D": 0.3611}
        self.dtype = torch.float32
        if model_management.should_use_fp16():
            self.dtype = torch.float16
        self.weight_path = {
            "Flux1.D": os.path.join(self.local_dir, "flux_resizer.pt"),
            "SDXL": os.path.join(self.local_dir, "sdxl_resizer.pt"),
            "SD 1.x": os.path.join(self.local_dir, "sd15_resizer.pt"),
        }
        self.version = "none"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "version": (["Flux1.D", "SDXL", "SD 1.x"],),
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

    def upscale(self, latent, version, upscale):
        device = model_management.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        if version != self.version:
            self.model = LatentResizer.load_model(
                self.weight_path[version], device, self.dtype
            )
            self.version = version

        self.model.to(device=device)
        scale_factor = self.scale_factor[version]
        latent_out = self.model(scale_factor * samples, scale=upscale) / scale_factor

        if self.dtype != torch.float32:
            latent_out = latent_out.to(dtype=torch.float32)

        latent_out = latent_out.to(device="cpu")

        self.model.to(device=model_management.vae_offload_device())
        return ({"samples": latent_out},)
