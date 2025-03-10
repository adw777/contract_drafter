import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

def load_contract_dataset(json_file):
    """Load and prepare the contract dataset."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prepared_data = []
    for item in data:
        conversation = [
            {"role": "user", "content": f"Generate an Indian legal contract for: {item['contract_type']}"},
            {"role": "assistant", "content": f"Let me think about this:\n{item['chain_of_thought']}\n\nHere's the contract:\n{item['raw_response']}"}
        ]
        prepared_data.append({"conversations": conversation})
    
    return Dataset.from_list(prepared_data)

def prepare_model_and_tokenizer(max_seq_length=2048):
    """Initialize the model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit", # unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    
    return model, tokenizer

def prepare_dataset(dataset, tokenizer):
    """Prepare dataset with formatting function closure."""
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(
            convo, 
            tokenize=False,
            add_generation_prompt=False
        ) for convo in convos]
        return {"text": texts}
    
    return dataset.map(formatting_prompts_func, batched=True)

def main():
    print("Loading dataset...")
    dataset = load_contract_dataset("data.json")
    
    print("\nInitializing model and tokenizer...")
    model, tokenizer = prepare_model_and_tokenizer()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nPreparing dataset...")
    dataset = prepare_dataset(dataset, tokenizer)
    
    print("\nSetting up trainer...")
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=2,
        max_steps=40,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    print("\nStarting training...")
    trainer_stats = trainer.train()
    
    print("\nSaving model...")
    model.save_pretrained("finetuned_contract_model")
    tokenizer.save_pretrained("finetuned_contract_model")
    
    print(f"\nTraining complete! Time taken: {trainer_stats.metrics['train_runtime']} seconds")

if __name__ == "__main__":
    main()