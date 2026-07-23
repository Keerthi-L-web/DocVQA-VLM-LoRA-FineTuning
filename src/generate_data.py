import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
import src.config as config

def generate_synthetic_data(num_samples=1000, split_ratios=(0.8, 0.1, 0.1)):
    """
    Generates synthetic document images (like invoices/memos) and Q&A pairs.
    """
    print(f"Generating {num_samples} synthetic document images...")
    
    img_dir = config.DATA_DIR / "images"
    os.makedirs(img_dir, exist_ok=True)
    
    data = []
    
    # Try to load a default font, otherwise fallback to default PIL font
    try:
        # This works on Windows if arial is present
        font = ImageFont.truetype("arial.ttf", 24)
        title_font = ImageFont.truetype("arialbd.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    companies = ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Stark Ind."]
    
    for i in tqdm(range(num_samples)):
        # Create a blank white image (document)
        img = Image.new('RGB', (800, 1000), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        # Generate random document content
        company = random.choice(companies)
        invoice_id = f"INV-{random.randint(1000, 9999)}"
        date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        total = round(random.uniform(10.0, 1000.0), 2)
        
        # Draw text on image
        d.text((50, 50), company, fill=(0, 0, 0), font=title_font)
        d.text((50, 120), "INVOICE", fill=(50, 50, 50), font=title_font)
        d.text((50, 200), f"Invoice Number: {invoice_id}", fill=(0, 0, 0), font=font)
        d.text((50, 250), f"Date: {date}", fill=(0, 0, 0), font=font)
        d.text((50, 350), "Items:", fill=(0, 0, 0), font=font)
        
        y_offset = 400
        for _ in range(random.randint(1, 5)):
            item_price = round(random.uniform(5.0, 100.0), 2)
            d.text((100, y_offset), f"- Product XYZ .... ${item_price}", fill=(0, 0, 0), font=font)
            y_offset += 40
            
        d.text((50, y_offset + 50), f"Total Amount: ${total}", fill=(200, 0, 0), font=title_font)
        
        # Save image
        img_filename = f"doc_{i:04d}.png"
        img_path = img_dir / img_filename
        img.save(img_path)
        
        # Generate Q&A pairs for this document
        qa_pairs = [
            {"question": "What is the company name?", "answer": company},
            {"question": "What is the invoice number?", "answer": invoice_id},
            {"question": "What is the date of the invoice?", "answer": date},
            {"question": "What is the total amount?", "answer": f"${total}"}
        ]
        
        # Select one random QA pair for the dataset sample to keep it simple
        # or we could keep all. Let's keep one random one for this sample.
        qa = random.choice(qa_pairs)
        
        data.append({
            "image_path": str(img_path.relative_to(config.DATA_DIR)),
            "question": qa["question"],
            "answers": [qa["answer"]]  # List format for VQA evaluation compatibility
        })

    # Shuffle and split data
    random.seed(42)
    random.shuffle(data)
    
    train_idx = int(num_samples * split_ratios[0])
    val_idx = train_idx + int(num_samples * split_ratios[1])
    
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    
    # Save splits
    for split_name, split_data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
        with open(config.DATA_DIR / f"{split_name}.json", "w") as f:
            json.dump(split_data, f, indent=4)
            
    print(f"Data generation complete! Saved to {config.DATA_DIR}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    generate_synthetic_data(num_samples=1000)
