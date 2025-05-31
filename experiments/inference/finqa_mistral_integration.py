import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from static_function_parser import StaticFunctionParser
import warnings
warnings.filterwarnings('ignore')

class FinQAMistralProcessor:
    """
    Processes FinQA dataset using Mistral 7B with static function calling.
    """
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.function_parser = StaticFunctionParser()
        
    def load_model(self):
        """Load Mistral model and tokenizer with explicit slow tokenizer."""
        print(f"🚀 Loading {self.model_name}...")
        
        # Load tokenizer with explicit slow tokenizer to avoid issues
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,  # This prevents the tokenizer issues
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model and tokenizer loaded successfully!")
        
    def create_finqa_prompt(self, question, context=""):
        """
        Create a prompt specifically designed for FinQA problems.
        """
        available_functions = ", ".join(self.function_parser.functions.keys())
        
        prompt = f"""<s>[INST] You are a financial calculation expert. Given a financial question, you must solve it using mathematical functions and output EXACTLY ONE nested function call.

Available functions: {available_functions}

Context: {context}

Question: {question}

Instructions:
1. Analyze the financial problem carefully
2. Use nested function calls when needed (e.g., percentage(multiply(price, quantity), total))
3. Output your answer as: FUNCTION: your_function_call_here
4. Do not add explanations after the function call

FUNCTION: [/INST]"""
        
        return prompt
    
    def generate_response(self, prompt, max_new_tokens=256):
        """Generate response using Mistral with simple tokenization."""
        print(f"🤖 Generating response...")
        
        # Simple tokenization to avoid fast tokenizer issues
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Lower temperature for more deterministic math
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
        response = full_response[len(prompt_text):].strip()
        
        return response
    
    def process_finqa_question(self, question, context=""):
        """
        Process a single FinQA question end-to-end.
        """
        print(f"📊 Processing FinQA question: {question[:100]}...")
        
        # Create prompt
        prompt = self.create_finqa_prompt(question, context)
        print(f"📝 Prompt created (length: {len(prompt)} chars)")
        
        # Generate response
        response = self.generate_response(prompt)
        print(f"🤖 Model response: {response}")
        
        # Parse and execute function call
        result = self.function_parser.execute_from_text(response)
        
        return {
            "question": question,
            "context": context,
            "prompt": prompt,
            "model_response": response,
            "function_result": result,
            "final_answer": result.get("result", "Error") if result["success"] else "Error"
        }
    
    def load_finqa_data(self, file_path):
        """Load FinQA dataset from JSON file."""
        print(f"📂 Loading FinQA data from {file_path}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} questions")
        return data
    
    def evaluate_batch(self, questions, max_questions=5):
        """
        Evaluate a batch of FinQA questions.
        """
        results = []
        
        for i, question_data in enumerate(questions[:max_questions]):
            print(f"\n{'='*60}")
            print(f"Question {i+1}/{min(len(questions), max_questions)}")
            print(f"{'='*60}")
            
            # Extract question and context from FinQA format
            if isinstance(question_data, dict):
                question = question_data.get('question', str(question_data))
                context = question_data.get('context', '')
                gold_answer = question_data.get('answer', 'Unknown')
            else:
                question = str(question_data)
                context = ''
                gold_answer = 'Unknown'
            
            # Process the question
            result = self.process_finqa_question(question, context)
            result['gold_answer'] = gold_answer
            result['question_id'] = i
            
            # Print results
            print(f"🎯 Gold Answer: {gold_answer}")
            print(f"🤖 Predicted Answer: {result['final_answer']}")
            
            if result['function_result']['success']:
                print(f"✅ Function executed successfully")
                print(f"📞 Function call: {result['function_result']['function_call']}")
            else:
                print(f"❌ Function execution failed: {result['function_result']['error']}")
            
            results.append(result)
        
        return results

# Example usage for Colab
def create_colab_cells():
    """
    Returns the code cells for Colab notebook.
    """
    
    cells = {
        "cell_1": """
# Cell 1: Install Dependencies
!pip install torch transformers accelerate
""",
        
        "cell_2": """
# Cell 2: Upload the static_function_parser.py and finqa_mistral_integration.py files
# Or copy the code directly into cells

# Copy the StaticFunctionParser class here if not uploading files
""",
        
        "cell_3": """
# Cell 3: Initialize the processor
from finqa_mistral_integration import FinQAMistralProcessor

processor = FinQAMistralProcessor()
processor.load_model()
""",
        
        "cell_4": """
# Cell 4: Test with a simple FinQA-style question
test_question = "A company's revenue increased from $100 million to $120 million. What is the percentage increase?"

result = processor.process_finqa_question(test_question)
print("Final Result:", result['final_answer'])
""",
        
        "cell_5": """
# Cell 5: Test with more complex nested calculations
complex_question = "If a company has 3 products with revenues of $50M, $30M, and $20M respectively, and the total market size is $500M, what is the company's market share percentage?"

result = processor.process_finqa_question(complex_question)
print("Final Result:", result['final_answer'])
""",
        
        "cell_6": """
# Cell 6: Load and process FinQA dataset (if you have the data file)
# Upload your FinQA JSON file first

# Example questions for testing
sample_questions = [
    {
        "question": "What is 25% of $400?",
        "answer": 100
    },
    {
        "question": "If revenue grew from $200M to $250M, what is the growth rate?",
        "answer": 25
    },
    {
        "question": "Calculate the average of 10, 20, and 30.",
        "answer": 20
    }
]

results = processor.evaluate_batch(sample_questions)

# Print summary
correct = 0
for result in results:
    if abs(float(result['final_answer']) - float(result['gold_answer'])) < 0.01:
        correct += 1

print(f"\\nAccuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
"""
    }
    
    return cells

if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing FinQA Mistral Integration")
    
    # This would be run in Colab
    processor = FinQAMistralProcessor()
    
    # Test without loading model (for development)
    test_question = "What is 20% of 150?"
    
    # Test the function parser directly
    parser = StaticFunctionParser()
    test_output = "To calculate 20% of 150, I need to use the percentage function. FUNCTION: percentage(20, 100)"
    
    result = parser.execute_from_text(test_output)
    print(f"Test result: {result}") 