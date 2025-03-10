from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TextStreamer
import json
from datetime import datetime

def load_model():
    print("Loading model and tokenizer...")
    # Load tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained("finetuned_contract_model")
    
    # Load model from local directory
    model = AutoModelForCausalLM.from_pretrained(
        "finetuned_contract_model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def generate_contract(model, tokenizer, contract_type):
    # Create prompt
    prompt = f"Generate an Indian legal contract for: {contract_type}"
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nGenerating {contract_type}...")
    print("-" * 50)
    
    # Generate text without streaming for saving
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    response = generated_text[len(prompt):].strip()
    
    # Print the output
    print(response)
    print("\n" + "=" * 50)
    
    return {
        "contract_type": contract_type,
        "response": response
    }

def save_to_json(contracts, filename=None):
    # Generate filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_contracts_{timestamp}.json"
    
    # Save contracts to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(contracts, f, indent=2, ensure_ascii=False)
    
    print(f"\nContracts saved to {filename}")

def main():
    # Free memory
    torch.cuda.empty_cache()
    
    # Load model
    model, tokenizer = load_model()
    
    # Test cases
    test_contracts = [
        "Employment Agreement Type 1",
        "Service Agreement between Freelancer and Company",
        "Non-Disclosure Agreement Type 2"
    ]
    
    # Store generated contracts
    generated_contracts = []
    
    # Generate each contract
    for contract_type in test_contracts:
        contract_data = generate_contract(model, tokenizer, contract_type)
        generated_contracts.append(contract_data)
    
    # Save all contracts to JSON
    save_to_json(generated_contracts)

if __name__ == "__main__":
    main()