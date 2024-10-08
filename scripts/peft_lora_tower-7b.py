# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import os
import argparse


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def parse_args():

    parser = argparse.ArgumentParser(description="PEFT finetune llama3 model")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Unbabel/TowerInstruct-7B-v0.2",
        help="",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default="data/en-de_ro_es.emea-iate.train.inst-tower.json",
        help="",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='models/tower-7b-en_de_es_ro-emea_1epoch_4bit_ft4',
        help="",
    )

    parser.add_argument(
        "--n_epoch",
        type=int,
        default=1,
        help="",
    )

    parser.add_argument(
        "--train_bs",
        type=int,
        default=2,
        help="",
    )

    parser.add_argument(
        "--grad_acc",
        type=int,
        default=2,
        help="",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="",
    )

    parser.add_argument(
        "--w_steps",
        type=float,
        default=0.03,
        help="",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="",
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="",
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="",
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="",
    )
    
    args = parser.parse_args()

    return args

def main():

    model_id = "Unbabel/TowerInstruct-7B-v0.2"
    #"/home/mrios/workspace/controlled_LLM/llama-2-7b-hf" 
    max_length = 512 #1024
    #max_target_length = 512
    code2lang = {
    "de": "German",
    "fr": "French",
    "en": "English",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ro": "Romanian",
    "es": "Spanish"
    }
    #source_code = 'en'
    #target_code = 'de'
    data_files = {}
    #"data/en-de_ro_es.emea-iate.train.split1.json
    data_files["train"] = "data/en-de_ro_es.emea-iate.train.inst-tower.json" #"data/en-de.emea.20k.train.json"
    #"data/en-de.emea.5k.train2.json"
    #https://arxiv.org/pdf/2312.12740.pdf trainig size 20k
    output_dir = 'models/tower-7b-en_de_es_ro-emea_1epoch_4bit_ft4'
    train_bs = 2
    grad_acc = 2
    lr = 2e-5
    w_steps = 0.03
    n_epoch = 1
    lr_scheduler_type = "cosine"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, 
            quantization_config=bnb_config, 
            device_map={"": 0})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig( #BEST MODEL!!
        r=64, 
        lora_alpha=16, 
        lora_dropout=0.1, #0.05
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    )   

 

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    print(model)

    def preprocess_parallel_function(examples):
        
        inputs = [ex['text'] for ex in examples["translation"]]    
           
        
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
        return model_inputs
    

    
    data = load_dataset("json", data_files=data_files)
   
    column_names = data["train"].column_names
    #print(column_names)
    data = data.map(preprocess_parallel_function,
                    batched=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=train_bs,
            gradient_accumulation_steps=grad_acc,
            warmup_ratio=w_steps,
            lr_scheduler_type=lr_scheduler_type,
            num_train_epochs=n_epoch,
            learning_rate=lr,
            fp16=True,
            save_total_limit=1,
            save_strategy="epoch",
            output_dir=output_dir,
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  
    trainer.train()

    trainer.save_model(output_dir)
  
    tokenizer.save_pretrained(output_dir)



if __name__ == '__main__':
    main()
