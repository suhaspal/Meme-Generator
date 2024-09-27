import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

def generate_image(prompt, model_path, output_dir, num_inference_steps=50, guidance_scale=7.5, width=512, height=512):
    print(f"Loading model from {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32, safety_checker=None)
    
    device = "cpu"
    print(f"Using device: {device}")
    pipe = pipe.to(device)

    print(f"Generating image with prompt: '{prompt}'")
    image = pipe(
        prompt, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        width=width,
        height=height
    ).images[0]
    
    # Check if the image is all black
    if np.array(image).sum() == 0:
        print("Warning: Generated image is all black!")
    
    # Create a valid filename from the prompt
    valid_filename = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
    valid_filename = valid_filename[:30]  # Limit filename length
    output_filename = f"generated_meme_{valid_filename}.png"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full output path
    output_path = os.path.join(output_dir, output_filename)
    
    image.save(output_path)
    print(f"Image saved to {output_path}")
    
    # Print image statistics
    img_array = np.array(image)
    print(f"Image shape: {img_array.shape}")
    print(f"Image min value: {img_array.min()}")
    print(f"Image max value: {img_array.max()}")
    print(f"Image mean value: {img_array.mean()}")

    image.show()

if __name__ == "__main__":
    model_path = "trained-comp-vis-model"
    output_dir = "generated_memes"
    
    while True:
        prompt = input("Enter a prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        generate_image(
            prompt, 
            model_path, 
            output_dir, 
            num_inference_steps=200,  # Increased from 50 to 00
            guidance_scale=8.5,       # Slightly increased from 7.5
            width=768,                # Increased from default 512
            height=768                # Increased from default 512
        )

print("Thank you for using the Meme Generator!")