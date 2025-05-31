import torch
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class MistralFinQAInference:
    """
    Pure Mistral 7B inference on FinQA dataset with function calling prompts.
    Handles complete FinQA format: pre_text, table, post_text, question.
    """
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load Mistral model and tokenizer."""
        print(f"🚀 Loading {self.model_name}...")
        print("⏳ This may take a few minutes...")
        
        # Load tokenizer with explicit slow tokenizer to avoid issues
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,  # Prevents tokenizer issues
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
        return True
    
    def format_table_for_prompt(self, table_data):
        """Convert table data to a readable format for the prompt."""
        if not table_data:
            return "No table data provided."
        
        # Format table as markdown-style
        formatted_table = "TABLE:\n"
        for i, row in enumerate(table_data):
            if i == 0:
                # Header row
                formatted_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                formatted_table += "|" + "---|" * len(row) + "\n"
            else:
                # Data rows
                formatted_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return formatted_table
    
    def format_finqa_context(self, finqa_item):
        """Format complete FinQA context including pre_text, table, post_text."""
        context_parts = []
        
        # Add pre-text
        if 'pre_text' in finqa_item and finqa_item['pre_text']:
            context_parts.append("CONTEXT BEFORE TABLE:")
            for text in finqa_item['pre_text']:
                context_parts.append(text.strip())
            context_parts.append("")
        
        # Add table
        if 'table' in finqa_item and finqa_item['table']:
            context_parts.append(self.format_table_for_prompt(finqa_item['table']))
            context_parts.append("")
        
        # Add post-text
        if 'post_text' in finqa_item and finqa_item['post_text']:
            context_parts.append("CONTEXT AFTER TABLE:")
            for text in finqa_item['post_text']:
                context_parts.append(text.strip())
        
        return "\n".join(context_parts)
        
    def create_finqa_function_calling_prompt(self, finqa_item):
        """
        Create an optimized few-shot prompt for FinQA with complete context and function calling instructions.
        """
        
        # Extract question
        if 'qa' in finqa_item and 'question' in finqa_item['qa']:
            question = finqa_item['qa']['question']
        else:
            question = finqa_item.get('question', 'No question found')
        
        # Format complete context
        context = self.format_finqa_context(finqa_item)
        
        # Extract table metadata for better analysis
        table_info = ""
        if 'table' in finqa_item and finqa_item['table']:
            table = finqa_item['table']
            table_info = f"Table has {len(table)} rows and {len(table[0]) if table else 0} columns.\n"
            if len(table) > 1:
                table_info += f"Headers: {', '.join(str(cell) for cell in table[0])}\n"
                table_info += f"Sample data types: {', '.join('numeric' if any(char.isdigit() for char in str(cell)) else 'text' for cell in table[1] if len(table) > 1)}"
        
        available_functions = "add, subtract, multiply, divide, percentage, percentage_change, ratio, average, sum, max, min, round"
        
        prompt = f"""<s>[INST] You are a financial analysis expert. Analyze financial data and solve questions using EXACTLY ONE nested function call.

AVAILABLE FUNCTIONS: {available_functions}

IMPORTANT INSTRUCTIONS:
1. First analyze the table structure and identify relevant numbers
2. Extract specific numerical values from the table/text
3. Create ONE nested function call that solves the problem
4. Use proper numerical values (e.g., 5735 for $5735, 1000000 for $1M)
5. Output ONLY: FUNCTION: your_nested_function_call
6. Do not add explanations, steps, or extra text

TABLE METADATA:
{table_info}

FINANCIAL DATA:
{context}

EXAMPLES:

Example 1:
Question: What is the percentage change in net revenue from 2014 to 2015?
Table shows: 2014 net revenue = $5735, 2015 net revenue = $5829
Analysis: Need to calculate percentage change from 5735 to 5829
FUNCTION: percentage_change(5735, 5829)

Example 2:
Question: What is the average of quarterly revenues if Q1=$8M, Q2=$9.5M, Q3=$7.2M, Q4=$10.1M?
Analysis: Need to average four quarterly values in millions
FUNCTION: average(8000000, 9500000, 7200000, 10100000)

Example 3:
Question: If retail electric price increased by $187M and volume/weather added $95M, what is the total increase?
Analysis: Need to add two positive changes
FUNCTION: add(187, 95)

Example 4:
Question: What percentage of total revenue does the largest division represent if divisions have $15M, $22M, and $18M?
Analysis: Find max division and calculate its percentage of total
FUNCTION: percentage(max(15000000, 22000000, 18000000), add(15000000, add(22000000, 18000000)))

CURRENT QUESTION: {question}

INSTRUCTIONS:
1. Identify the specific numbers needed from the table/text above
2. Determine what calculation is required
3. Create ONE nested function call using the identified numbers
4. Output format: FUNCTION: your_function_call

FUNCTION: [/INST]"""
        
        return prompt, question, context
    
    def create_function_calling_prompt(self, question, context=""):
        """
        Legacy method for backwards compatibility with simple questions.
        """
        available_functions = "add, subtract, multiply, divide, percentage, percentage_change, ratio, average, sum, max, min, round"
        
        prompt = f"""<s>[INST] You are a financial calculation expert. Analyze the question and solve it using EXACTLY ONE nested function call.

AVAILABLE FUNCTIONS: {available_functions}

IMPORTANT INSTRUCTIONS:
1. Analyze the given information carefully
2. Extract numerical values needed for calculation
3. Create ONE nested function call that solves the problem
4. Use proper numerical values (e.g., 100000000 for $100M)
5. Output ONLY: FUNCTION: your_nested_function_call
6. Do not add explanations or extra text

CONTEXT: {context}

EXAMPLES:

Example 1:
Question: What is the percentage increase from $100M to $125M?
Analysis: Calculate percentage change from 100000000 to 125000000
FUNCTION: percentage_change(100000000, 125000000)

Example 2:
Question: What is 25% of $2 million?
Analysis: Calculate 25 percent of 2000000
FUNCTION: percentage(25, 2000000)

Example 3:
Question: What is the average of $5M, $6.2M, $4.8M, and $7.1M?
Analysis: Average four values in millions
FUNCTION: average(5000000, 6200000, 4800000, 7100000)

CURRENT QUESTION: {question}

INSTRUCTIONS:
1. Identify the numbers and operation needed
2. Create ONE nested function call
3. Output format: FUNCTION: your_function_call

FUNCTION: [/INST]"""
        
        return prompt
    
    def generate_response(self, prompt, max_new_tokens=256, temperature=0.3):
        """Generate response using Mistral with configurable parameters."""
        print(f"🤖 Generating response (max_tokens={max_new_tokens}, temp={temperature})...")
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096  # Increased for longer FinQA contexts
        )
        
        print(f"📝 Input tokens: {inputs.shape[1]}")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
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
        
        generation_time = time.time() - start_time
        
        print(f"⏱️ Generation time: {generation_time:.2f}s")
        print(f"📤 Output tokens: {outputs.shape[1] - inputs.shape[1]}")
        
        return response
    
    def process_finqa_item(self, finqa_item, show_prompt=False):
        """
        Process a complete FinQA item with pre_text, table, post_text, and question.
        """
        print(f"\n{'='*80}")
        print(f"📊 PROCESSING FINQA ITEM")
        print(f"{'='*80}")
        
        # Create prompt with full FinQA context
        prompt, question, context = self.create_finqa_function_calling_prompt(finqa_item)
        
        print(f"❓ Question: {question}")
        print(f"📋 Context length: {len(context)} characters")
        
        # Show gold answer if available
        if 'qa' in finqa_item and 'answer' in finqa_item['qa']:
            gold_answer = finqa_item['qa']['answer']
            print(f"🎯 Gold Answer: {gold_answer}")
        
        # Show expected program if available
        if 'qa' in finqa_item and 'program' in finqa_item['qa']:
            expected_program = finqa_item['qa']['program']
            print(f"📈 Expected Program: {expected_program}")
        
        if show_prompt:
            print(f"\n📝 FULL PROMPT:")
            print("-" * 40)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("-" * 40)
        
        # Generate response
        response = self.generate_response(prompt)
        
        print(f"\n🤖 MODEL RESPONSE:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        return {
            "question": question,
            "context": context,
            "prompt": prompt,
            "response": response,
            "gold_answer": finqa_item.get('qa', {}).get('answer', 'Unknown'),
            "expected_program": finqa_item.get('qa', {}).get('program', 'Unknown'),
            "finqa_id": finqa_item.get('id', 'Unknown'),
            "prompt_length": len(prompt),
            "response_length": len(response)
        }
    
    def process_finqa_question(self, question, context="", show_prompt=False):
        """
        Legacy method for backwards compatibility with simple questions.
        """
        print(f"\n{'='*80}")
        print(f"📊 PROCESSING SIMPLE QUESTION")
        print(f"{'='*80}")
        print(f"❓ Question: {question}")
        if context:
            print(f"📋 Context: {context}")
        
        # Create prompt
        prompt = self.create_function_calling_prompt(question, context)
        
        if show_prompt:
            print(f"\n📝 PROMPT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
        
        # Generate response
        response = self.generate_response(prompt)
        
        print(f"\n🤖 MODEL RESPONSE:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        return {
            "question": question,
            "context": context,
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response)
        }
    
    def load_finqa_data(self, file_path):
        """Load actual FinQA dataset from JSON file."""
        print(f"📂 Loading FinQA data from {file_path}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} FinQA items")
        return data
    
    def load_finqa_sample_data(self):
        """Load sample FinQA-style questions for testing (legacy compatibility)."""
        
        sample_questions = [
            {
                "question": "What is the percentage increase in revenue from $2.5 million to $3.2 million?",
                "context": "Company reported revenue of $2.5M last year and $3.2M this year.",
                "expected_function": "percentage_change(2500000, 3200000)",
                "gold_answer": 28.0
            },
            {
                "question": "If a company has three divisions with revenues of $15M, $22M, and $18M, what percentage of the total does the largest division represent?",
                "context": "Division A: $15M, Division B: $22M, Division C: $18M",
                "expected_function": "percentage(max(15000000, 22000000, 18000000), add(15000000, 22000000, 18000000))",
                "gold_answer": 40.0
            },
            {
                "question": "What is the average quarterly revenue if Q1=$5M, Q2=$6.2M, Q3=$4.8M, and Q4=$7.1M?",
                "context": "Quarterly revenues: Q1=$5M, Q2=$6.2M, Q3=$4.8M, Q4=$7.1M",
                "expected_function": "average(5000000, 6200000, 4800000, 7100000)",
                "gold_answer": 5.775
            }
        ]
        
        return sample_questions
    
    def run_batch_inference(self, questions=None, max_questions=None, show_prompts=False, use_finqa_format=True):
        """
        Run inference on a batch of questions.
        
        Args:
            questions: List of FinQA items or simple questions
            max_questions: Maximum number to process
            show_prompts: Whether to show full prompts
            use_finqa_format: Whether to expect full FinQA format or simple questions
        """
        if questions is None:
            if use_finqa_format:
                # Try to load real FinQA data
                try:
                    questions = self.load_finqa_data('data/finqa/test.json')
                    print("📊 Using real FinQA dataset")
                except FileNotFoundError:
                    print("❌ Real FinQA data not found, using sample questions")
                    questions = self.load_finqa_sample_data()
                    use_finqa_format = False
            else:
                questions = self.load_finqa_sample_data()
        
        if max_questions:
            questions = questions[:max_questions]
        
        print(f"\n🚀 RUNNING BATCH INFERENCE")
        print(f"📊 Processing {len(questions)} questions")
        print(f"🤖 Model: {self.model_name}")
        print(f"📋 Format: {'Full FinQA' if use_finqa_format else 'Simple Questions'}")
        
        results = []
        total_start_time = time.time()
        
        for i, question_data in enumerate(questions, 1):
            print(f"\n{'🔥' * 20}")
            print(f"QUESTION {i}/{len(questions)}")
            print(f"{'🔥' * 20}")
            
            if use_finqa_format:
                # Process as full FinQA item
                result = self.process_finqa_item(
                    question_data, 
                    show_prompt=show_prompts
                )
            else:
                # Process as simple question (legacy)
                if isinstance(question_data, dict):
                    question = question_data.get('question', str(question_data))
                    context = question_data.get('context', '')
                    expected_function = question_data.get('expected_function', 'N/A')
                    gold_answer = question_data.get('gold_answer', 'Unknown')
                else:
                    question = str(question_data)
                    context = ''
                    expected_function = 'N/A'
                    gold_answer = 'Unknown'
                
                result = self.process_finqa_question(
                    question, 
                    context, 
                    show_prompt=show_prompts
                )
                
                # Add metadata for simple questions
                result.update({
                    'expected_function': expected_function,
                    'gold_answer': gold_answer
                })
            
            result['question_id'] = i
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'🏁' * 20}")
        print(f"BATCH INFERENCE COMPLETE")
        print(f"{'🏁' * 20}")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"⚡ Average time per question: {total_time/len(questions):.2f}s")
        print(f"📊 Total questions processed: {len(results)}")
        
        return results
    
    def save_results(self, results, filename="mistral_finqa_results.json"):
        """Save inference results to a JSON file."""
        
        print(f"\n💾 Saving results to {filename}...")
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'question_id': result['question_id'],
                'question': result['question'],
                'response': result['response'],
                'gold_answer': result.get('gold_answer', 'Unknown'),
                'expected_program': result.get('expected_program', 'Unknown'),
                'finqa_id': result.get('finqa_id', 'Unknown'),
                'prompt_length': result['prompt_length'],
                'response_length': result['response_length']
            }
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Results saved to {filename}")
        return filename

def main():
    """Main function to run the inference pipeline."""
    
    print("🔥 MISTRAL 7B FINQA INFERENCE PIPELINE")
    print("=" * 60)
    
    # Initialize the inference engine
    inference = MistralFinQAInference()
    
    # Load model
    if not inference.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Run inference on real FinQA data
    print(f"\n📋 Processing FinQA dataset...")
    
    # Run batch inference with real FinQA format
    results = inference.run_batch_inference(
        max_questions=3,  # Process first 3 questions
        show_prompts=False,  # Set to True to see full prompts
        use_finqa_format=True  # Use full FinQA format with context
    )
    
    # Save results
    filename = inference.save_results(results)
    
    print(f"\n🎉 INFERENCE COMPLETE!")
    print(f"📄 Results saved to: {filename}")
    print(f"🔍 Check the JSON file for detailed model outputs")

if __name__ == "__main__":
    main() 