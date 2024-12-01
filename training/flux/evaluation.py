#!/usr/bin/env python
import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from latent_resizer import LatentResizer
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms
from tqdm import tqdm


def psnr(x, ref, maxg=2):
    mse = torch.mean(torch.square(x - ref))
    return 20 * torch.log10(maxg / torch.sqrt(mse))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, size):
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError("Dataset path does not exist")
        self.images = list(self.path.iterdir())
        self.num_images = len(self.images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        example = {}
        image = Image.open(self.images[index % self.num_images]).convert("RGB")
        image = self.image_transforms(image)
        return image


def collate_fn(images):
    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent resizer evaluation")
    parser.add_argument(
        "--test_path",
        required=True,
        type=str,
        help="Test images",
    )
    parser.add_argument(
        "--vae_path",
        required=True,
        type=str,
        help="VAE path",
    )
    parser.add_argument(
        "--resizer_path",
        required=True,
        type=str,
        help="Resizer weight path",
    )
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        default="cuda",
        help="Torch device",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 precision",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--resolution",
        default=256,
        type=int,
        help="Image resolution",
    )
    parser.add_argument(
        "--scale",
        default=2.0,
        type=float,
        required=True,
        help="Resize scale",
    )
    parser.add_argument(
        "--resizer_only",
        action="store_true",
        help="Only evaluate resizer",
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    scale_factor = 0.3611

    if args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype=dtype)
    vae.eval()

    resizer = LatentResizer.load_model(args.resizer_path, device, dtype)

    # LPIPS is always in float32 because of nans in float16
    lpips_fn = lpips.LPIPS(net="vgg").to(device=device, dtype=torch.float32)

    dataset = ImageDataset(args.test_path, args.resolution)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    elapsed_vae_list = []
    elapsed_resizer_list = []
    elapsed_latent_list = []

    mse_vae_list = []
    mse_resizer_list = []
    mse_latent_list = []

    lpips_vae_list = []
    lpips_resizer_list = []
    lpips_latent_list = []

    psnr_vae_list = []
    psnr_resizer_list = []
    psnr_latent_list = []

    ssim_vae_list = []
    ssim_resizer_list = []
    ssim_latent_list = []

    try:
        with torch.inference_mode():
            for images in tqdm(dataloader):
                images = images.to(device=device, dtype=dtype)
                images_upscaled = torch.nn.functional.interpolate(
                    images, scale_factor=args.scale, mode="bilinear"
                )
                latents = vae.encode(images).latent_dist.sample()
                del images

                # Resizer
                start_resizer = torch.cuda.Event(enable_timing=True)
                end_resizer = torch.cuda.Event(enable_timing=True)
                start_resizer.record()
                resized = (
                    resizer(scale_factor * latents, scale=args.scale) / scale_factor
                )
                end_resizer.record()
                torch.cuda.synchronize()
                resizer_elapsed = start_resizer.elapsed_time(end_resizer)

                decoded_resized = vae.decode(resized)[0]
                del resized

                elapsed_resizer_list.append(resizer_elapsed)

                mse_resizer = torch.nn.functional.mse_loss(
                    decoded_resized, images_upscaled
                )
                mse_resizer_list.extend(mse_resizer.cpu().numpy().flatten())

                lpips_resizer = lpips_fn(
                    decoded_resized.float(), images_upscaled.float()
                )

                lpips_resizer_list.extend(lpips_resizer.cpu().numpy().flatten())

                psnr_resized = psnr(decoded_resized, images_upscaled)
                psnr_resizer_list.extend(psnr_resized.cpu().numpy().flatten())

                ssim_resized = ssim(
                    0.5 * (decoded_resized + 1),
                    0.5 * (images_upscaled + 1),
                    data_range=1,
                    size_average=True,
                )
                ssim_resizer_list.append(ssim_resized.cpu())

                if not args.resizer_only:
                    start_vae = torch.cuda.Event(enable_timing=True)
                    end_vae = torch.cuda.Event(enable_timing=True)

                    # VAE decode -> upscale -> encode
                    start_vae.record()
                    decoded_img = vae.decode(latents)[0]
                    img_upscaled = torch.nn.functional.interpolate(
                        decoded_img, scale_factor=args.scale, mode="bilinear"
                    )
                    vae_encoded = vae.encode(img_upscaled).latent_dist.sample()
                    end_vae.record()
                    torch.cuda.synchronize()
                    vae_elapsed = start_vae.elapsed_time(end_vae)

                    # Scale latent
                    start_latent = torch.cuda.Event(enable_timing=True)
                    end_latent = torch.cuda.Event(enable_timing=True)
                    start_latent.record()
                    resized_latent = torch.nn.functional.interpolate(
                        latents, scale_factor=args.scale, mode="bilinear"
                    )
                    end_latent.record()
                    torch.cuda.synchronize()
                    latent_elapsed = start_latent.elapsed_time(end_latent)

                    elapsed_vae_list.append(vae_elapsed)
                    elapsed_latent_list.append(latent_elapsed)

                    # Decode latents and calculate LPIPS and MSE
                    decoded_vae = vae.decode(vae_encoded)[0]
                    decoded_latent = vae.decode(resized_latent)[0]

                    mse_vae = torch.nn.functional.mse_loss(decoded_vae, images_upscaled)
                    mse_latent = torch.nn.functional.mse_loss(
                        decoded_latent, images_upscaled
                    )

                    mse_vae_list.extend(mse_vae.cpu().numpy().flatten())
                    mse_latent_list.extend(mse_latent.cpu().numpy().flatten())

                    lpips_vae = lpips_fn(decoded_vae.float(), images_upscaled.float())
                    lpips_latent = lpips_fn(
                        decoded_latent.float(), images_upscaled.float()
                    )

                    lpips_vae_list.extend(lpips_vae.cpu().numpy().flatten())
                    lpips_latent_list.extend(lpips_latent.cpu().numpy().flatten())

                    psnr_vae = psnr(decoded_vae, images_upscaled)
                    psnr_latent = psnr(decoded_latent, images_upscaled)

                    psnr_vae_list.extend(psnr_vae.cpu().numpy().flatten())
                    psnr_latent_list.extend(psnr_latent.cpu().numpy().flatten())

                    ssim_vae = ssim(
                        0.5 * (decoded_vae + 1),
                        0.5 * (images_upscaled + 1),
                        data_range=1,
                        size_average=True,
                    )
                    ssim_latent = ssim(
                        0.5 * (decoded_latent + 1),
                        0.5 * (images_upscaled + 1),
                        data_range=1,
                        size_average=True,
                    )

                    ssim_vae_list.append(ssim_vae.cpu())
                    ssim_latent_list.append(ssim_latent.cpu())
    finally:
        print("Batch size", args.batch_size)
        if not args.resizer_only:
            print("Elapsed VAE", np.mean(elapsed_vae_list), "ms")
            print("Elapsed latent upscale", np.mean(elapsed_latent_list), "ms")
        print("Elapsed resizer", np.mean(elapsed_resizer_list), "ms")

        if not args.resizer_only:
            print("MSE VAE", np.mean(mse_vae_list))
            print("MSE latent upscale", np.mean(mse_latent_list))
        print("MSE resizer", np.mean(mse_resizer_list))

        if not args.resizer_only:
            print("LPIPS VAE", np.mean(lpips_vae_list))
            print("LPIPS latent upscale", np.mean(lpips_latent_list))
        print("LPIPS resizer", np.mean(lpips_resizer_list))

        if not args.resizer_only:
            print("PSNR VAE", np.mean(psnr_vae_list))
            print("PSNR latent upscale", np.mean(psnr_latent_list))
        print("PSNR resizer", np.mean(psnr_resizer_list))

        if not args.resizer_only:
            print("SSIM VAE", np.mean(ssim_vae_list))
            print("SSIM latent upscale", np.mean(ssim_latent_list))
        print("SSIM resizer", np.mean(ssim_resizer_list))
