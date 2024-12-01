#!/usr/bin/env python
import argparse
import io
import os
import random
from collections import defaultdict

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from latent_resizer import LatentResizer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm


def init_dataset(dataset_path, size=512):
    shards = []
    if type(dataset_path) not in (list, tuple):
        dataset_path = [dataset_path]
    for path in dataset_path:
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if full_path.endswith(".tar"):
                shards.append(full_path)
    print(f"{len(shards)} shards")

    def preprocess(sample):
        k = [k for k in sample.keys() if k in ["jpg", "png"]]
        if len(k) == 0:
            raise ValueError("Dataset images should be in jpg or png format")
        k = k[0]
        img = sample[k]

        img = Image.open(io.BytesIO(img))
        if not img.mode == "RGB":
            img = img.convert("RGB")

        pil_image = img

        image_transforms = transforms.Compose(
            [
                transforms.RandomCrop(size, pad_if_needed=True, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        img = image_transforms(img)
        examples = {}
        examples["img"] = img
        return examples

    dataset = (
        wds.WebDataset(shards, handler=wds.warn_and_continue, shardshuffle=True)
        .repeat()
        .shuffle(256)
        .map(preprocess)
    )
    return dataset


def collate_fn(examples):
    imgs = [example["img"] for example in examples]
    imgs = torch.stack(imgs)

    scale = random.uniform(1.0, 2.1)
    size = [int(round(imgs.shape[-2] * scale)), int(round(imgs.shape[-1] * scale))]
    size[0] -= size[0] % 8
    size[1] -= size[1] % 8
    imgs_scaled = F.interpolate(imgs, size=size, mode="bilinear")

    if scale < 1:
        batch = {
            "img_input": imgs_scaled,
            "img_target": imgs,
        }
    else:
        batch = {
            "img_input": imgs,
            "img_target": imgs_scaled,
        }

    return batch


def calculate_loss(
    model,
    batch,
    vae,
    lpips,
    dtype=torch.float16,
    mse_weight=1,
    lpips_weight=0.1,
    mse_latent_weight=0.01,
):
    img_input = batch["img_input"].to(args.device, dtype=dtype)
    img_target = batch["img_target"].to(args.device, dtype=dtype)
    latent_input = (
        vae.config.scaling_factor * vae.encode(img_input).latent_dist.sample()
    )
    latent_target = (
        vae.config.scaling_factor * vae.encode(img_target).latent_dist.sample()
    )
    size = latent_target.shape[-2:]
    with torch.autocast(args.device, dtype=dtype, enabled=dtype != torch.float32):
        resized = model(latent_input, size=size)
    mse_latent = F.mse_loss(resized, latent_target)
    logs = {"mse_latent": mse_latent.detach().cpu().item()}
    decoded = vae.decode(resized / vae.config.scaling_factor)[0]
    mse = F.mse_loss(decoded, img_target)
    logs["mse"] = mse
    loss = mse_weight * mse + mse_latent_weight * mse_latent
    if lpips_weight > 0:
        ploss = lpips(decoded, img_target).mean()
        logs["lpips"] = ploss.detach().cpu().item()
        loss = loss + lpips_weight * ploss
    logs["loss"] = loss.detach().cpu().item()
    return loss, logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent interpolate trainer")
    parser.add_argument(
        "--train_path",
        required=True,
        action="append",
        help="Training data path for VAE latents. Webdataset format.",
    )
    parser.add_argument(
        "--test_path",
        default=None,
        required=False,
        action="append",
        help="Test data path for VAE latents. Webdataset format.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to VAE",
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=1000,
        required=False,
        help="Test interval",
    )
    parser.add_argument(
        "--test_batches",
        type=int,
        default=10,
        required=False,
        help="Number of test batches",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="flux_resizer.pt",
        required=False,
        help="Output filename",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of steps to train",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save model every this step",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="CPU workers",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Droput rate",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
        help="Gradient clipping",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--init_weights",
        type=str,
        default=None,
        help="Resume training from weights file",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 precision",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for VAE",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./data/runs",
        help="Tensorboard log directory",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    vae_dtype = torch.float32
    if args.fp16:
        vae_dtype = torch.float16

    print("Loading VAE")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype=vae_dtype)
    vae.config.scaling_factor = 0.3611

    vae.train()
    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()
    lpips_fn = lpips.LPIPS(net="vgg").to(device=device, dtype=vae_dtype)

    if args.init_weights:
        model = LatentResizer.load_model(
            args.init_weights,
            device=args.device,
            dropout=args.dropout,
            dtype=torch.float32,
        )
    else:
        model = LatentResizer(dropout=args.dropout).to(args.device)

    print("Loading datasets")
    train_dataset = init_dataset(args.train_path, size=args.resolution)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    if args.test_path:
        test_dataset = init_dataset(args.test_path, size=args.resolution)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    print("Loading training pipeline")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, total_iters=200
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[20]
    )
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(params, "Parameters")
    writer = SummaryWriter(comment="resizer", log_dir=args.log_dir)
    model.train()
    epoch = 0
    step = 0
    progress_bar = tqdm(range(args.steps))
    progress_bar.set_description("Steps")
    train_fn = lambda batch: calculate_loss(model, batch, vae, lpips_fn, vae_dtype)

    while step < args.steps:
        epoch += 1
        for batch in train_dataloader:
            if batch["img_input"].shape == batch["img_target"].shape:
                continue
            step += 1
            loss, logs = train_fn(batch)
            l = loss.detach().cpu().item()
            for k in logs.keys():
                writer.add_scalar("{}/train".format(k), logs[k], step)
            progress_bar.set_postfix(loss=round(l, 2), lr=scheduler.get_last_lr()[0])
            scaler.scale(loss).backward()
            if 0:
                total_norm = 0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)
                print("norm", total_norm)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            progress_bar.update(1)
            scheduler.step()
            if step >= args.steps:
                break
            if (step % args.save_steps) == 0:
                base, ext = os.path.splitext(args.output_filename)
                save_filename = f"{base}-{step}{ext}"
                torch.save(model.state_dict(), save_filename)
            if args.test_path and (step % args.test_steps) == 0:
                test_batches = 0
                test_logs = defaultdict(float)
                test_loss = 0
                model.eval()
                for batch in test_dataloader:
                    with torch.inference_mode():
                        _, logs = loss, logs = train_fn(batch)
                    test_batches += 1
                    for k in logs.keys():
                        test_logs[k] += logs[k]
                    if test_batches >= args.test_batches:
                        break
                model.train()
                for k in test_logs.keys():
                    writer.add_scalar(
                        "{}/test".format(k), test_logs[k] / test_batches, step
                    )

    torch.save(model.state_dict(), args.output_filename)
    print("Model saved")
