from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("axondendriteplus/contract_drafter-Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "axondendriteplus/contract_drafter-Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate contract
prompt = "Generate an Indian legal contract for: Employment Agreement"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
