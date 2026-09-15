import sys
import os
import io
import base64
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image as PILImage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.u_net import UNet
from model.noise_schedule import NoiseSchedule
from transformers import CLIPTokenizer, CLIPTextModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
timesteps = 1000

script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_dir, '..', 'model', 'unet_checkpoint.pt')

model = UNet(in_channels=3, base_channels=64).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

schedule = NoiseSchedule(timesteps=timesteps, device=device)

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_text_model.eval()

@torch.no_grad()
def encode_captions(captions, max_length=32):
    tokens = clip_tokenizer(
        list(captions), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    ).to(device)
    return clip_text_model(**tokens).last_hidden_state

@torch.no_grad()
def sample(prompt, image_size=64, batch_size=1, channels=3, guidance_scale=7.5):
    text_embed = encode_captions([prompt] * batch_size)
    null_embed = encode_captions([""] * batch_size)

    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    for t_step in reversed(range(schedule.timesteps)):
        t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

        pred_cond = model(x, t, text_embed)
        pred_uncond = model(x, t, null_embed)
        predicted_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        alpha = schedule.alphas[t_step]
        alpha_bar = schedule.alphas_cumprod[t_step]
        beta = schedule.betas[t_step]

        noise = torch.randn_like(x) if t_step > 0 else torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
        ) + torch.sqrt(beta) * noise

    return x


class PromptRequest(BaseModel):
    caption: str

@app.post('/generate')
async def generate_image(request: PromptRequest):
    samples = sample(request.caption)
    samples = (samples.clamp(-1, 1) + 1) / 2

    img_array = (samples[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    pil_img = PILImage.fromarray(img_array)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {'image' : f'data:image/png;base64, {img_base64}'}