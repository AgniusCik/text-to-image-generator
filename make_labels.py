'''
import os
import pandas as pd

data_dir = 'images/train'
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

rows = []
for cls in classes:
    folder = os.path.join(data_dir, cls)
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append({'filename': fname, 'label': cls})

df = pd.DataFrame(rows)

df.to_csv('labels.csv', index=False)
'''

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

data_dir = 'images/train'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

def caption_image(img_path):
    image = Image.open(img_path).convert('RGB')
    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output[0], skip_special_tokens=True)

def hybrid_caption(img_path, label):
    base = caption_image(img_path)
    return f'{base}, a type of {label}'

rows = []
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# gather all (path, class) pairs first so tqdm knows the total
tasks = []
for cls in classes:
    folder = os.path.join(data_dir, cls)
    for img in os.scandir(folder):
        tasks.append((img.path, img.name, cls))

for img_path, img_name, cls in tqdm(tasks, desc="Captioning images"):
    caption = hybrid_caption(img_path, cls)
    rel_path = os.path.join(cls, img_name)
    rows.append({'filename': rel_path, 'caption': caption})

df = pd.DataFrame(rows)
df.to_csv('captions.csv', index=False)
print(f"Saved {len(df)} captions to captions.csv")