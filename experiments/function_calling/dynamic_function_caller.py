import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import requests
from finqa_function_caller import FunctionCaller

class DynamicFunctionCaller:
    """Handles dynamic function calling with Mistral 7B Instruct."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.function_caller = FunctionCaller()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.conversation_history = []
        
    def format_function_schema(self) -> str:
        """Format the available functions as a schema for the model."""
        functions = {
            "calculator": "Basic arithmetic operations (e.g., 2+2)",
            "calendar": "Date formatting and calculations",
            "percentage": "Calculate percentage (e.g., 50% of 100)",
            "percentage_change": "Calculate percentage change between two values (e.g., 1000|1200 for change from 1000 to 1200)",
            "average": "Compute average of numbers",
            "sum": "Sum a list of numbers",
            "difference": "Compute difference between numbers",
            "ratio": "Compute ratio between numbers",
            "growth_rate": "Calculate growth rate between two values (e.g., 1000|1200 for growth from 1000 to 1200)",
            "compound_growth": "Calculate compound growth rate over multiple periods (e.g., 1000,1200,1500 for values over 3 periods)",
            "moving_average": "Calculate moving average over a window (e.g., 100,200,300,400|2 for 2-period moving average)",
            "standard_deviation": "Calculate standard deviation of a list of numbers",
            "variance": "Calculate variance of a list of numbers",
            "median": "Calculate median of a list of numbers",
            "mode": "Calculate mode of a list of numbers",
            "range": "Calculate range (max - min) of a list of numbers",
            "quartile": "Calculate quartile (1-4) of a list of numbers (e.g., 100,200,300,400|2 for second quartile)"
        }
        
        schema = "Available functions:\n"
        for name, desc in functions.items():
            schema += f"- {name}: {desc}\n"
        return schema
    
    def format_prompt(self, user_input: str) -> str:
        """Format the prompt with conversation history and function schema."""
        prompt = f"""<s>[INST] You are a helpful AI assistant that can use functions to help solve problems.
When you need to use a function, respond in the following JSON format:
{{
    "thought": "Your reasoning about what to do next",
    "function": "function_name",
    "arguments": "function_arguments"
}}

{self.format_function_schema()}

Previous conversation:
{self.format_conversation_history()}

User: {user_input}
[/INST]</s>"""
        return prompt
    
    def format_conversation_history(self) -> str:
        """Format the conversation history for the prompt."""
        if not self.conversation_history:
            return "No previous conversation."
        
        history = ""
        for entry in self.conversation_history:
            if entry["role"] == "user":
                history += f"User: {entry['content']}\n"
            elif entry["role"] == "assistant":
                history += f"Assistant: {entry['content']}\n"
            elif entry["role"] == "function":
                history += f"Function ({entry['name']}): {entry['result']}\n"
        return history
    
    def parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse the model's response to extract function call information."""
        try:
            # Find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                return {"thought": response, "function": None, "arguments": None}
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"thought": response, "function": None, "arguments": None}
    
    def execute_function(self, function_name: str, arguments: str) -> Any:
        """Execute the specified function with the given arguments."""
        if function_name in self.function_caller.functions:
            return self.function_caller.functions[function_name](arguments)
        return f"Error: Unknown function {function_name}"
    
    def process_response(self, response: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Process the model's response and execute any function calls."""
        function_results = []
        
        if response["function"]:
            result = self.execute_function(response["function"], response["arguments"])
            function_results.append({
                "name": response["function"],
                "arguments": response["arguments"],
                "result": result
            })
            
            # Add function result to conversation history
            self.conversation_history.append({
                "role": "function",
                "name": response["function"],
                "result": str(result)
            })
            
            return str(result), function_results
        
        return response["thought"], function_results
    
    def generate_response(self, user_input: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response using the model and handle function calls."""
        # Add user input to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Format and generate prompt
        prompt = self.format_prompt(user_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse and process response
        parsed_response = self.parse_model_response(response)
        final_response, function_results = self.process_response(parsed_response)
        
        # Add assistant response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response
        })
        
        return final_response, function_results

def main():
    try:
        # Initialize the dynamic function caller
        caller = DynamicFunctionCaller()
        
        # Example usage
        test_input = "Calculate the percentage increase from 1000 to 1200"
        print(f"\nUser: {test_input}")
        
        response, function_results = caller.generate_response(test_input)
        print(f"\nAssistant: {response}")
        
        if function_results:
            print("\nFunction Results:")
            for result in function_results:
                print(f"Function: {result['name']}")
                print(f"Arguments: {result['arguments']}")
                print(f"Result: {result['result']}")
                print("---")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 