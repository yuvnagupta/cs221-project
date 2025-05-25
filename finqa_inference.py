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
import pandas as pd

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
    """Load Mistral 7B model and tokenizer."""
    print("Loading Mistral 7B model and tokenizer...")
    model_name = "mistralai/Mistral-7B-v0.1"
    
    os.makedirs("./models", exist_ok=True)
    download_model_with_retry(model_name)
    
    model_path = f"./models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto"  # This helps with memory management
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
    """Format table for display with clear structure and alignment for LLM processing using pandas."""
    if not table or not table[0]:
        return ""
    
    # Convert table to pandas DataFrame
    df = pd.DataFrame(table[1:], columns=table[0])
    
    # Format the output
    formatted_rows = []
    
    # Add table description
    formatted_rows.append("Table Structure:")
    formatted_rows.append(f"Number of columns: {len(df.columns)}")
    formatted_rows.append(f"Number of rows: {len(df)}")
    formatted_rows.append("")
    
    # Add column information
    formatted_rows.append("Column Information:")
    for col in df.columns:
        # Get sample values and determine data type
        sample_values = df[col].dropna().head(3).tolist()
        if all(str(v).replace('.', '').replace('-', '').replace('$', '').replace(',', '').isdigit() for v in sample_values):
            dtype = "Numeric"
            # Add summary statistics for numeric columns
            stats = df[col].astype(float).describe()
            formatted_rows.append(f"- {col}:")
            formatted_rows.append(f"  Type: {dtype}")
            formatted_rows.append(f"  Min: {stats['min']:.2f}")
            formatted_rows.append(f"  Max: {stats['max']:.2f}")
            formatted_rows.append(f"  Mean: {stats['mean']:.2f}")
        else:
            dtype = "Text"
            formatted_rows.append(f"- {col}:")
            formatted_rows.append(f"  Type: {dtype}")
            formatted_rows.append(f"  Sample values: {', '.join(str(v) for v in sample_values)}")
    
    # Add formatted table
    formatted_rows.append("\nFormatted Table:")
    formatted_rows.append(df.to_string(index=False))
    
    # Add value extraction hints
    formatted_rows.append("\nValue Extraction Hints:")
    for idx, row in df.iterrows():
        formatted_rows.append(f"Row {idx+1}: {row.to_dict()}")
    
    # Add summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        formatted_rows.append("\nNumeric Column Statistics:")
        formatted_rows.append(df[numeric_cols].describe().to_string())
    
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
    """Generate answer using Mistral 7B."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,  # Added for better generation
        repetition_penalty=1.1,  # Added to reduce repetition
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
        
        # Simulate an LLM response that should trigger function calling
        simulated_response = """
        To calculate the percentage increase in revenue from Q1 to Q2 2024, we need to:
        1. Find the revenue values from the table
        2. Calculate the percentage change
        
        From the table:
        Q1 2024 Revenue: $1000000
        Q2 2024 Revenue: $1200000
        
        Let's calculate the percentage increase:
        [PERCENTAGE]1000000 of 1200000[/PERCENTAGE]
        
        Therefore, the revenue increased by 20% from Q1 to Q2 2024.
        """
        
        print("\nSimulated LLM Response:")
        print(simulated_response)
            
        # Process answer with function caller
        print("\nProcessing answer with function caller...")
        final_answer, function_results = process_answer_with_functions(simulated_response, function_caller)
        
        print("\nFinal Answer:")
        print(final_answer)
        print("\nFunction Results:")
        for func_name, args, result in function_results:
            print(f"Function: {func_name}")
            print(f"Arguments: {args}")
            print(f"Result: {result}")
            print("---")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 