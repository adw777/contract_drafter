import requests
import json
from typing import Dict, List
import time
import re
from datetime import datetime
import os

class ContractGenerator:
    def __init__(self, model_name="deepseek-r1:14b"):
        self.base_url = "http://localhost:11434"
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from Ollama API: {response.text}")
            
        return response.json()["response"]

def get_contract_types() -> List[str]:
    """Generate 500 different contract types."""
    contract_types = [
        # Business Contracts
        *[f"Business Partnership Agreement Type {i}" for i in range(1, 21)],
        *[f"Franchise Agreement Variation {i}" for i in range(1, 21)],
        *[f"Distribution Agreement Type {i}" for i in range(1, 21)],
        *[f"Supply Chain Contract Version {i}" for i in range(1, 21)],
        
        # Employment Contracts
        *[f"Employment Agreement Type {i}" for i in range(1, 21)],
        *[f"Consultant Contract Version {i}" for i in range(1, 21)],
        *[f"Internship Agreement Type {i}" for i in range(1, 21)],
        *[f"Remote Work Contract Version {i}" for i in range(1, 21)],
        
        # Real Estate
        *[f"Property Lease Agreement Type {i}" for i in range(1, 21)],
        *[f"Real Estate Purchase Contract {i}" for i in range(1, 21)],
        *[f"Commercial Property Agreement {i}" for i in range(1, 21)],
        *[f"Construction Contract Type {i}" for i in range(1, 21)],
        
        # Technology
        *[f"Software Development Agreement {i}" for i in range(1, 21)],
        *[f"IT Services Contract Type {i}" for i in range(1, 21)],
        *[f"Technology License Agreement {i}" for i in range(1, 21)],
        *[f"SaaS Agreement Version {i}" for i in range(1, 21)],
        
        # Services
        *[f"Professional Services Agreement {i}" for i in range(1, 21)],
        *[f"Maintenance Contract Type {i}" for i in range(1, 21)],
        *[f"Consulting Services Agreement {i}" for i in range(1, 21)],
        *[f"Outsourcing Contract Version {i}" for i in range(1, 21)],
        
        # Miscellaneous
        *[f"Confidentiality Agreement Type {i}" for i in range(1, 21)],
        *[f"Joint Venture Contract {i}" for i in range(1, 21)],
        *[f"Service Level Agreement {i}" for i in range(1, 21)],
        *[f"Manufacturing Agreement {i}" for i in range(1, 21)]
    ]
    
    # Ensure we have exactly 500 contracts
    return contract_types[:500]

def parse_response(response: str) -> Dict:
    """Extract thinking process and contract content from response."""
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if not think_match:
        raise ValueError("No thinking process found in response")
    
    chain_of_thought = think_match.group(1).strip()
    raw_response = response.split('</think>')[-1].strip()
    
    return {
        "chain_of_thought": chain_of_thought,
        "raw_response": raw_response
    }

def generate_training_examples(start_index=0, batch_size=None) -> List[Dict]:
    contract_types = get_contract_types()
    if batch_size:
        contract_types = contract_types[start_index:start_index + batch_size]
    
    generator = ContractGenerator()
    examples = []
    total = len(contract_types)
    
    # Create directory for saving progress
    os.makedirs("backup", exist_ok=True)
    
    for idx, contract_type in enumerate(contract_types, 1):
        prompt = f"""Generate an Indian legal contract for: {contract_type}

        Please follow this format exactly:
        1. Start with a thinking process in <think> tags explaining your reasoning
        2. Then provide the actual contract content
        3. The contract should follow Indian legal requirements
        4. Include all necessary clauses and sections

        Make it detailed and legally accurate."""

        try:
            print(f"\nGenerating contract {idx + start_index} of {total + start_index}: {contract_type}")
            response = generator.generate(prompt)
            
            parsed = parse_response(response)
            example = {
                "contract_type": contract_type,
                "chain_of_thought": parsed["chain_of_thought"],
                "raw_response": parsed["raw_response"]
            }
            examples.append(example)
            
            # Save backup every 10 examples
            if idx % 10 == 0:
                backup_file = f"backup/contracts_backup_{start_index + idx}.json"
                save_to_json(examples, backup_file)
                print(f"Backup saved to {backup_file}")
            
            # Add delay to avoid overwhelming the API
            time.sleep(2)
            
        except Exception as e:
            print(f"Error generating {contract_type}: {e}")
            # Save error log
            with open("error_log.txt", "a") as f:
                f.write(f"{datetime.now()}: Error with {contract_type}: {str(e)}\n")
            continue

    return examples

def save_to_json(examples: List[Dict], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(examples)} examples to {filename}")

def main():
    try:
        batch_size = 50  # Process in batches of 50
        total_contracts = 500
        all_examples = []
        
        for start_idx in range(0, total_contracts, batch_size):
            print(f"\nStarting batch {start_idx//batch_size + 1} of {total_contracts//batch_size}")
            examples = generate_training_examples(start_idx, batch_size)
            all_examples.extend(examples)
            
            # Save intermediate results
            intermediate_file = f"indian_legal_contracts_dataset_batch_{start_idx//batch_size + 1}.json"
            save_to_json(all_examples, intermediate_file)
        
        # Save final complete dataset
        save_to_json(all_examples, "indian_legal_contracts_dataset_complete.json")
        
        # Print final statistics
        print("\nGeneration Complete!")
        print(f"Total contracts generated: {len(all_examples)}")
        print(f"Final file saved as: indian_legal_contracts_dataset_complete.json")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()