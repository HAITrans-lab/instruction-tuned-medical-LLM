# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import transformers
import os
import re
import numpy as np


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

def main():

    model_id = "google/flan-t5-large"
    #"/home/mrios/workspace/controlled_LLM/llama-2-7b-hf" #"NousResearch/Llama-2-7b-hf" #"EleutherAI/gpt-neox-20b"
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
    data_files["train"] = "data/en-de_ro_es.emea-iate.train.split1.json" #"data/en-de.emea.20k.train.json"
    #"data/en-de.emea.5k.train2.json"
    #https://arxiv.org/pdf/2312.12740.pdf trainig size 20k
    output_dir = 'models/flan-t5large__en-de_es_ro_emea_1epoch_ft'
    train_bs = 6
    grad_acc = 1
    lr = 2e-4
    w_steps = 0.03
    n_epoch = 1
    lr_scheduler_type = "linear"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, 
            quantization_config=bnb_config, 
            device_map={"": 0})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config =LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

   
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    print(model)

    metric = None

    def preprocess_parallel_function(examples):
        #src_lang = code2lang[source_code]
        #tgt_lang = code2lang[target_code]
        texts = [re.split(r'(\nTarget: |\nGerman: |\nSpanish: |\nRomanian: )', ex['text']) for ex in examples["translation"]]
        #print(texts)
        inputs = [t[0] for t in texts]
        targets = [t[2] for t in texts]      
       
        #exit(1)
        model_inputs = tokenizer(inputs, max_length=max_length, padding=False, truncation=True)

        # Setup the tokenizer for targets
        #with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, padding=False, truncation=True)
      

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

    def postprocess_text(preds, labels):
      preds = [pred.strip() for pred in preds]
      labels = [[label.strip()] for label in labels]

      return preds, labels

    def compute_metrics(eval_preds, ignore_pad_token_for_loss=False):
      preds, labels = eval_preds
      if isinstance(preds, tuple):
        preds = preds[0]
      decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
      # Replace -100 in the labels as we can't decode them.
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
      decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
      # Some simple post-processing
      decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
      result = metric.compute(predictions=decoded_preds, references=decoded_labels)
      prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
      result = {'bleu' : result['score']}
      result["gen_len"] = np.mean(prediction_lens)
      result = {k: round(v, 4) for k, v in result.items()}
      return result
    
    
    data = load_dataset("json", data_files=data_files)
   
    column_names = data["train"].column_names
    #print(column_names)
    data = data.map(preprocess_parallel_function,
                    batched=True)

    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "right"
    #tokenizer.pad_token_id
    label_pad_token_id = -100

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=train_bs,
            gradient_accumulation_steps=grad_acc,
            warmup_ratio=w_steps,
            lr_scheduler_type=lr_scheduler_type,
            num_train_epochs=n_epoch,
            learning_rate=lr,
            fp16=False, #
            save_total_limit=2,
            save_strategy="epoch",
            output_dir=output_dir,
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=label_pad_token_id),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    trainer.save_model(output_dir)

    #output_dir = os.path.join(output_dir, "final_checkpoint")
    #trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == '__main__':
    main()
