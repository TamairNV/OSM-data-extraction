import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image

print("Loading datasets...")
training = json.load(open('training_dataSet.json'))
validation = json.load(open('validation_dataSet.json'))

# 1. Safely target Apple Silicon
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Training on device: {device}")
from peft import PeftModel
# 2. Native Apple Silicon Load (No bitsandbytes!)
print("Loading base model in native bfloat16...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    torch_dtype=torch.bfloat16,
).to(device)

model = PeftModel.from_pretrained(base_model, 'fpv_model', is_trainable=True)

# 3. Enable Gradient Checkpointing (Saves ~60% VRAM during training)
model.gradient_checkpointing_enable()

processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')

"""
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)
"""

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

print("Starting Training...")
model.train()

EPOCHS = 2
ACCUMULATION_STEPS = 4

for epoch in range(EPOCHS):
    total_loss = 0
    for idx, x in enumerate(training):
        # 5. Restrict image resolution to prevent VRAM spikes!
        image = Image.open(x['image']).convert("RGB")
        image.thumbnail((512, 512))

        prompt_text = x['conversations'][0]['content']
        label_text = x['conversations'][1]['content']

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": label_text}]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_only_text = processor.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_only_text], images=[image], return_tensors="pt")

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_inputs.input_ids.shape[1]] = -100
        inputs["labels"] = labels

        outputs = model(**inputs)
        loss = outputs.loss/ ACCUMULATION_STEPS

        loss.backward()
        if (idx + 1) % ACCUMULATION_STEPS == 0 or (idx + 1) == len(training):
            optimizer.step()
            optimizer.zero_grad()


        total_loss += (loss.item() * ACCUMULATION_STEPS)

        if idx % 50 == 0:
            print(f"Epoch {epoch + 1} | Step {idx}/{len(training)} | Loss: {(loss.item() * ACCUMULATION_STEPS):.4f}")

        # 6. Strict Mac VRAM Cleanup
        del loss, outputs, inputs, labels, prompt_inputs
        torch.mps.empty_cache()

    print(f"--- Epoch {epoch + 1} Average Training Loss: {total_loss / len(training):.4f} ---")

print("Starting Validation...")
model.eval()

with torch.no_grad():
    val_loss_total = 0
    for idx, x in enumerate(validation):
        image = Image.open(x['image']).convert("RGB")
        image.thumbnail((512, 512))

        prompt_text = x['conversations'][0]['content']
        label_text = x['conversations'][1]['content']

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": label_text}]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_only_text = processor.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_only_text], images=[image], return_tensors="pt")

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_inputs.input_ids.shape[1]] = -100
        inputs["labels"] = labels

        outputs = model(**inputs)
        val_loss = outputs.loss.item()
        val_loss_total += val_loss

        if idx % 50 == 0:
            print(f"Validation Step {idx}/{len(validation)} | Val Loss: {val_loss:.4f}")

        del outputs, inputs, labels, prompt_inputs
        torch.mps.empty_cache()

    print(f"Final Validation Loss: {val_loss_total / len(validation):.4f}")

print("Saving model weights...")
model.save_pretrained('fpv_model')
processor.save_pretrained('fpv_model')
print("Training Complete!")