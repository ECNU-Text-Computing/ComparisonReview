from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import os


def sft(venue: str) -> None:
    base_path = "../model/Qwen2.5-7B-Instruct"
    lora_path = f"../model/{venue}/Qwen2.5-7B-Instruct-lora"
    sft_path = f"../model/{venue}/Qwen2.5-7B-Instruct-sft"
    dataset_path = f"../data/{venue}/comparisons_sft"

    dataset = load_from_disk(dataset_path)
    dataset = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
          
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

        
    sft_config = SFTConfig(
        bf16=True,
        max_length=1024,
        num_train_epochs=1,
        learning_rate=5e-4,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=2,
        average_tokens_across_devices=False,
        save_strategy="no",
        save_total_limit=0,
        output_dir=lora_path,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()

                     
    trainer.save_model(lora_path)
                 
    print("Merging LoRA weights into base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(sft_path, safe_serialization=True)
    tokenizer.save_pretrained(sft_path)
    print("Merged model saved to ./model/Qwen2.5-7B-Instruct-sft")


if __name__ == "__main__":
    sft("iclr_2025")
