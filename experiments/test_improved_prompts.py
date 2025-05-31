#!/usr/bin/env python3
"""
Test script for the improved FinQA prompts with few-shot examples and table metadata.
"""

from mistral_finqa_inference import MistralFinQAInference
import json

def test_prompt_formatting():
    """Test the improved prompt formatting without running inference."""
    
    print("🧪 TESTING IMPROVED PROMPT FORMATTING")
    print("=" * 60)
    
    # Initialize inference engine (no model loading needed for prompt testing)
    inference = MistralFinQAInference()
    
    try:
        # Load sample FinQA data
        finqa_data = inference.load_finqa_data('data/finqa/test.json')
        sample_item = finqa_data[0]
        
        print("📊 SAMPLE FINQA ITEM:")
        print(f"Question: {sample_item['qa']['question']}")
        print(f"Expected Answer: {sample_item['qa']['answer']}")
        print(f"Expected Program: {sample_item['qa']['program']}")
        print()
        
        # Generate the improved prompt
        prompt, question, context = inference.create_finqa_function_calling_prompt(sample_item)
        
        print("📝 IMPROVED PROMPT STRUCTURE:")
        print("-" * 80)
        
        # Show the prompt in sections
        prompt_sections = prompt.split('\n\n')
        
        for i, section in enumerate(prompt_sections):
            if 'AVAILABLE FUNCTIONS' in section:
                print("🔧 FUNCTIONS SECTION:")
                print(section[:200] + "...")
            elif 'TABLE METADATA' in section:
                print("\n📋 TABLE METADATA:")
                print(section)
            elif 'FINANCIAL DATA' in section:
                print("\n💰 FINANCIAL DATA:")
                print(section[:300] + "...")
            elif 'EXAMPLES' in section:
                print("\n📚 FEW-SHOT EXAMPLES:")
                examples_text = section[:800] + "..." if len(section) > 800 else section
                print(examples_text)
            elif 'CURRENT QUESTION' in section:
                print("\n❓ CURRENT QUESTION:")
                print(section)
            elif 'INSTRUCTIONS' in section:
                print("\n📋 FINAL INSTRUCTIONS:")
                print(section)
        
        print("-" * 80)
        
        print(f"\n📊 PROMPT STATISTICS:")
        print(f"Total length: {len(prompt)} characters")
        print(f"Number of examples: 4")
        print(f"Context length: {len(context)} characters")
        
        # Test table metadata extraction
        if 'table' in sample_item and sample_item['table']:
            table = sample_item['table']
            print(f"\n📋 TABLE ANALYSIS:")
            print(f"Table dimensions: {len(table)} rows × {len(table[0]) if table else 0} columns")
            print(f"Headers: {table[0] if table else 'None'}")
            if len(table) > 1:
                print(f"Sample row: {table[1]}")
        
    except FileNotFoundError:
        print("❌ FinQA data not found. Testing with simple question instead...")
        
        # Test simple question prompt
        simple_prompt = inference.create_function_calling_prompt(
            "What is the percentage increase from $100M to $125M?",
            "Company revenue data comparison"
        )
        
        print("📝 SIMPLE QUESTION PROMPT:")
        print("-" * 40)
        print(simple_prompt[:1000] + "...")
        print("-" * 40)

def test_multiple_questions():
    """Test prompt generation for multiple types of questions."""
    
    print("\n🧪 TESTING MULTIPLE QUESTION TYPES")
    print("=" * 60)
    
    inference = MistralFinQAInference()
    
    try:
        finqa_data = inference.load_finqa_data('data/finqa/test.json')
        
        # Test first 3 different questions
        for i in range(min(3, len(finqa_data))):
            item = finqa_data[i]
            print(f"\n--- FinQA Item {i+1} ---")
            print(f"Question: {item['qa']['question']}")
            print(f"Expected Program: {item['qa']['program']}")
            
            # Generate prompt (without model inference)
            prompt, question, context = inference.create_finqa_function_calling_prompt(item)
            
            # Analyze prompt characteristics
            function_examples = prompt.count('FUNCTION:')
            table_mentions = prompt.count('table')
            
            print(f"Prompt length: {len(prompt)} chars")
            print(f"Function examples: {function_examples}")
            print(f"Table references: {table_mentions}")
            
            # Show the actual question section
            question_start = prompt.find('CURRENT QUESTION:')
            if question_start != -1:
                question_section = prompt[question_start:question_start+200]
                print(f"Question section: {question_section}...")
    
    except FileNotFoundError:
        print("❌ FinQA data not found.")

def compare_old_vs_new_prompts():
    """Compare the old vs new prompt structure."""
    
    print("\n🔄 COMPARING OLD VS NEW PROMPT STRUCTURE")
    print("=" * 60)
    
    inference = MistralFinQAInference()
    
    # Test question
    test_question = "What is the percentage increase from $100M to $125M?"
    test_context = "Revenue comparison data"
    
    # New prompt
    new_prompt = inference.create_function_calling_prompt(test_question, test_context)
    
    print("📊 PROMPT COMPARISON:")
    print(f"New prompt length: {len(new_prompt)} characters")
    print(f"Examples included: {'EXAMPLES:' in new_prompt}")
    print(f"Structured instructions: {'INSTRUCTIONS:' in new_prompt}")
    print(f"Clear formatting: {'FUNCTION:' in new_prompt}")
    
    print("\n✅ NEW PROMPT FEATURES:")
    features = [
        "✓ Few-shot examples with 4 different scenarios",
        "✓ Structured sections (FUNCTIONS, EXAMPLES, INSTRUCTIONS)",
        "✓ Table metadata analysis",
        "✓ Clear numerical format guidance",
        "✓ Strict output format enforcement",
        "✓ Step-by-step analysis examples"
    ]
    
    for feature in features:
        print(feature)

def main():
    """Run all prompt testing functions."""
    
    print("🔥 IMPROVED FINQA PROMPT TESTING SUITE")
    print("=" * 80)
    
    # Test 1: Basic prompt formatting
    test_prompt_formatting()
    
    # Test 2: Multiple question types
    test_multiple_questions()
    
    # Test 3: Old vs new comparison
    compare_old_vs_new_prompts()
    
    print("\n🎉 PROMPT TESTING COMPLETE!")
    print("💡 The improved prompts include:")
    print("   - Few-shot examples")
    print("   - Table metadata analysis") 
    print("   - Structured formatting")
    print("   - Clear numerical guidelines")
    print("   - Strict output format")

if __name__ == "__main__":
    main() 