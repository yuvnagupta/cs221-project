import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import requests
import time
import subprocess
from huggingface_hub import snapshot_download
from finqa_function_caller import FunctionCaller, format_prompt_with_functions

def download_finqa_dataset():
    """Download FinQA dataset if not present."""
    if not os.path.exists("data/finqa/test.json"):
        print("Downloading FinQA dataset...")
        os.makedirs("data/finqa", exist_ok=True)
        
        url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
        response = requests.get(url)
        if response.status_code == 200:
            with open("data/finqa/test.json", "wb") as f:
                f.write(response.content)
            print("Successfully downloaded FinQA test dataset")
        else:
            raise Exception(f"Failed to download FinQA dataset. Status code: {response.status_code}")

def download_model_with_retry(model_name, max_retries=3, retry_delay=10):
    """Download model with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to download model...")
            snapshot_download(
                repo_id=model_name,
                local_dir=f"./models/{model_name}",
                local_dir_use_symlinks=False,
                resume_download=True
            )
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to download model after {max_retries} attempts")

def load_model_and_tokenizer():
    """Load Phi-2 model and tokenizer."""
    print("Loading Phi-2 model and tokenizer...")
    model_name = "microsoft/phi-2"
    
    os.makedirs("./models", exist_ok=True)
    download_model_with_retry(model_name)
    
    model_path = f"./models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model and tokenizer loaded successfully")
    return model, tokenizer

def load_finqa_data(split="test"):
    """Load FinQA dataset."""
    if not os.path.exists(f"data/finqa/{split}.json"):
        download_finqa_dataset()
    
    print(f"Loading {split} dataset...")
    with open(f"data/finqa/{split}.json", "r") as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} examples")
    return data

def format_table(table):
    """Format table for display."""
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    formatted_rows = []
    for row in table:
        formatted_row = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
        formatted_rows.append(formatted_row)
    formatted_rows.insert(1, "-" * len(formatted_rows[0]))
    return "\n".join(formatted_rows)

def format_prompt(item):
    """Format the input prompt for the model with function calling instructions."""
    context = "\n".join(item["pre_text"] + item["post_text"])
    table_info = format_table(item["table"])
    question = item["qa"]["question"]
    
    return format_prompt_with_functions().format(
        context=context,
        table_info=table_info,
        question=question
    )

def generate_answer(model, tokenizer, prompt, max_new_tokens=512):
    """Generate answer using Phi-2."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    return answer

def process_answer_with_functions(answer, function_caller):
    """Process the answer using the function caller."""
    try:
        final_answer, function_results = function_caller.process_answer(answer)
        return final_answer, function_results
    except Exception as e:
        print(f"Error processing answer with functions: {str(e)}")
        return answer, []

def save_predictions(predictions, output_file):
    """Save predictions to a file."""
    print(f"Saving predictions to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print("Predictions saved successfully")

def download_evaluation_script():
    """Download evaluation script if not present."""
    if not os.path.exists("finqa/evaluate.py"):
        print("Downloading evaluation script...")
        os.makedirs("finqa", exist_ok=True)
        
        url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/evaluate.py"
        response = requests.get(url)
        if response.status_code == 200:
            with open("finqa/evaluate.py", "wb") as f:
                f.write(response.content)
            print("Successfully downloaded evaluation script")
        else:
            raise Exception(f"Failed to download evaluation script. Status code: {response.status_code}")

def run_evaluation(pred_file, gold_file):
    """Run the official FinQA evaluation script."""
    if not os.path.exists("finqa/evaluate.py"):
        download_evaluation_script()
    
    eval_script = "finqa/evaluate.py"
    cmd = f"python {eval_script} --pred_file {pred_file} --gold_file {gold_file}"
    
    print("Running evaluation...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("Evaluation Results:")
        print(result.stdout)
        if result.stderr:
            print("Evaluation Errors:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running evaluation: {e}")

def main():
    try:
        # Initialize function caller
        function_caller = FunctionCaller()
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Load test data
        test_data = load_finqa_data(split="test")
        
        # Run inference
        print("Running inference...")
        predictions = []
        
        for item in tqdm(test_data):
            # Format prompt with function calling instructions
            prompt = format_prompt(item)
            
            # Generate answer
            answer = generate_answer(model, tokenizer, prompt)
            
            # Process answer with function caller
            final_answer, function_results = process_answer_with_functions(answer, function_caller)
            
            predictions.append({
                "id": item["id"],
                "raw_answer": answer,
                "final_answer": final_answer,
                "function_results": [
                    {
                        "function": func_name,
                        "args": args,
                        "result": str(result)
                    }
                    for func_name, args, result in function_results
                ]
            })
        
        # Save predictions
        pred_file = "outputs/finqa_function_predictions.json"
        save_predictions(predictions, pred_file)
        
        # Run evaluation using the official FinQA evaluation script
        gold_file = "data/finqa/test.json"
        run_evaluation(pred_file, gold_file)
        
        print("Inference completed successfully!")
        print(f"Results saved to {pred_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 