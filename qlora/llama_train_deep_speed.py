#
# This is the script for training Llama 1B locally using Deepspe
# 
# deepspeed /home/yikaiyang/Desktop/NLP_Lora/qlora/llama_deep_speed.py --deepspeed --deepspeed-config deepspeed.json
#
#

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets, load_from_disk
from trl import SFTTrainer
import torch
import random
import numpy as np
from peft import LoraModel, LoraConfig
from evaluate import load
import math
from transformers import BitsAndBytesConfig
import torch
from peft import  get_peft_model

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


# Configure quantization
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
#            load_in_8bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
)

# LORA config
parameters = {
    "output_folder": 'llama_3_1_1B',
    "sequence_length": 256,
    "epochs": 1,
    "batch_size": 3,
    "learning_rate": 2e-4,
    "optimizer": 'adamw_torch',
    "lora_alpha": 16,
    "lora_rank": 64,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "gate_proj", "down_proj", "up_proj"],
    "lora_drop_out": 0.1
}

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=parameters["lora_rank"],
    lora_alpha=parameters["lora_alpha"],
    target_modules=parameters["lora_target_modules"],
    lora_dropout=parameters["lora_drop_out"],
    init_lora_weights=True
)

# Load general model
model_name = "unsloth/Llama-3.2-1B-bnb-4bit"
#model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = '<|finetune_right_pad_id|>'

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

# Load peft model
peft_model = get_peft_model(model, config)

# Load dataset
# Tokenize the dataset
def preprocess_function(examples):
    # Remove entries with -1
    return tokenizer(examples["text"], truncation=True, max_length=parameters['sequence_length'])
#datasets = load_dataset('yikaiyang/qlora_instruct')
datasets = load_from_disk('./data').shuffle()

subset_size = len(datasets) // 5
datasets = datasets.select(range(subset_size))
datasets_tokenized = datasets.map(preprocess_function, batched=True)

def train():
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./{parameters['output_folder']}/results",  # Directory to save model checkpoints
        evaluation_strategy="no",
        learning_rate=parameters['learning_rate'],
        per_device_train_batch_size=parameters['batch_size'],
        per_device_eval_batch_size=parameters['batch_size'],
        num_train_epochs=parameters['epochs'],
    #    weight_decay=parameters['weight_decay'],
    
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        optim=parameters['optimizer'],
        report_to="none",
    #    warmup_ratio=parameters['warmup_ratio'],
    )

    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=datasets_tokenized,
    #    max_seq_length=parameters['sequence_length'],
        args=training_args
    )

    # Train the model
    #trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Save model
    trainer.save_model(f"./{parameters['output_folder']}/model")

train()