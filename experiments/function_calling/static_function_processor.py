#!/usr/bin/env python3
"""
Process Mistral FinQA results by extracting function calls and executing them.
Creates a new file with numerical responses from the nested function caller.
"""

import json
import re
from static_function_parser import StaticFunctionParser


def clean_function_call(response_text):
    """
    Clean and extract the main function call from response text.
    Handle cases with explanations, multiple functions, etc.
    """
    # Remove backslashes that might be escaping underscores
    response_text = response_text.replace('\\_', '_')
    
    # Look for FUNCTION: pattern first
    function_match = re.search(r'FUNCTION:\s*(.+?)(?:\n|$|%%)', response_text, re.MULTILINE | re.IGNORECASE)
    if function_match:
        function_call = function_match.group(1).strip()
        
        # If there are multiple function calls on the same line, take the first one
        if '\n' in function_call:
            function_call = function_call.split('\n')[0]
        
        # Clean up any trailing explanations or comments
        if '%' in function_call:
            function_call = function_call.split('%')[0].strip()
        
        return function_call
    
    # Fallback: look for any function-like pattern
    func_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\))'
    matches = re.findall(func_pattern, response_text)
    if matches:
        return matches[0].strip()
    
    return ""


def process_complex_response(response_text, parser):
    """
    Handle complex responses that might have multiple function calls or explanations.
    Try to extract the most relevant function call.
    """
    # First try the standard extraction
    result = parser.execute_from_text(response_text)
    if result["success"]:
        return result
    
    # Try cleaning the function call first
    cleaned_call = clean_function_call(response_text)
    if cleaned_call:
        result = parser.execute_from_text(f"FUNCTION: {cleaned_call}")
        if result["success"]:
            return result
    
    # Try to find simple mathematical operations
    simple_patterns = [
        r'subtract\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
        r'add\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
        r'multiply\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
        r'divide\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
        r'percentage\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
    ]
    
    for pattern in simple_patterns:
        match = re.search(pattern, response_text)
        if match:
            func_name = pattern.split('\\')[0]
            args = match.groups()
            simple_call = f"{func_name}({', '.join(args)})"
            result = parser.execute_from_text(f"FUNCTION: {simple_call}")
            if result["success"]:
                return result
    
    # Return the original error result
    return result


def main():
    # Initialize the function parser
    parser = StaticFunctionParser()
    
    # Load the original results
    print("📁 Loading mistral_finqa_results-2.json...")
    with open('mistral_finqa_results-2.json', 'r') as f:
        original_results = json.load(f)
    
    print(f"📊 Processing {len(original_results)} entries...")
    
    # Process each entry
    processed_results = []
    success_count = 0
    error_count = 0
    
    for i, entry in enumerate(original_results):
        print(f"\n🔢 Processing entry {i+1}/{len(original_results)} (ID: {entry['question_id']})")
        print(f"📝 Original response: {entry['response'][:100]}...")
        
        # Try to execute the function call
        execution_result = process_complex_response(entry['response'], parser)
        
        # Create new entry
        new_entry = entry.copy()
        
        if execution_result["success"]:
            new_entry["response"] = str(execution_result["result"])
            new_entry["execution_success"] = True
            new_entry["original_response"] = entry["response"]
            new_entry["function_call_extracted"] = execution_result.get("function_call", "")
            print(f"✅ Numerical result: {execution_result['result']}")
            success_count += 1
        else:
            new_entry["response"] = f"ERROR: {execution_result['error']}"
            new_entry["execution_success"] = False
            new_entry["original_response"] = entry["response"]
            new_entry["function_call_extracted"] = execution_result.get("function_call", "")
            new_entry["error_details"] = execution_result["error"]
            print(f"❌ Error: {execution_result['error']}")
            error_count += 1
        
        processed_results.append(new_entry)
    
    # Save the processed results
    output_filename = 'mistral_finqa_results_processed.json'
    print(f"\n💾 Saving processed results to {output_filename}...")
    
    with open(output_filename, 'w') as f:
        json.dump(processed_results, f, indent=2)
    
    # Print summary
    print(f"\n📈 Processing Summary:")
    print(f"   Total entries: {len(original_results)}")
    print(f"   ✅ Successful executions: {success_count}")
    print(f"   ❌ Failed executions: {error_count}")
    print(f"   📊 Success rate: {success_count/len(original_results)*100:.1f}%")
    print(f"\n🎉 Processed results saved to: {output_filename}")
    
    # Show some examples
    print(f"\n📋 Sample processed entries:")
    for i in range(min(3, len(processed_results))):
        entry = processed_results[i]
        print(f"\n   Entry {entry['question_id']}:")
        print(f"   Question: {entry['question'][:80]}...")
        print(f"   Original: {entry['original_response'][:60]}...")
        print(f"   Processed: {entry['response']}")
        print(f"   Success: {entry['execution_success']}")


if __name__ == "__main__":
    main() 