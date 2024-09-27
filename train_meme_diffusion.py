import time
import argparse
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

class MemeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir = data_dir
        self.size = size
        self.tokenizer = tokenizer
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = image.resize((self.size, self.size))
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1)
        return image

def main(args):
    # Set up device
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)

    unet.enable_gradient_checkpointing()

    # Prepare dataset and dataloader
    dataset = MemeDataset(args.train_data_dir, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Training loop
    # Set the maximum training time (in seconds)
    max_train_time = 60 * 60  # 60 minutes

    # Training loop
    start_time = time.time()
    step = 0
    progress_bar = tqdm(total=max_train_time, desc="Training")

    while time.time() - start_time < max_train_time:
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            latents = vae.encode(batch).latent_dist.sample()
            latents = latents * 0.18215

            # Get text embeddings
            text_input = tokenizer(args.placeholder_token, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            text_embeddings = text_encoder(text_input.input_ids)[0]

            # Predict the noise residual
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
            noisy_latents = noise + latents
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            step += 1

            # Update progress bar
            elapsed_time = time.time() - start_time
            progress_bar.update(elapsed_time - progress_bar.n)
            progress_bar.set_postfix(loss=loss.item(), step=step)

            if elapsed_time >= max_train_time:
                break

        if time.time() - start_time >= max_train_time:
            break

    progress_bar.close()
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Save the fine-tuned model
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
    )
    pipeline.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--train_data_dir", type=str, default="data/images", required=True)
    parser.add_argument("--learnable_property", type=str, default="style")
    parser.add_argument("--placeholder_token", type=str, default="<meme-style>")
    parser.add_argument("--initializer_token", type=str, default="style")
    parser.add_argument("--output_dir", type=str, default="trained-comp-vis-model", required=True)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)

    args = parser.parse_args()
    main(args)