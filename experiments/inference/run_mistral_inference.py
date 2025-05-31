#!/usr/bin/env python3
"""
Simple script to run Mistral 7B inference on FinQA questions.
Supports both full FinQA format (pre_text, table, post_text) and simple questions.
"""

from mistral_finqa_inference import MistralFinQAInference

def run_single_finqa_item():
    """Run inference on a single FinQA item from the dataset."""
    
    print("🚀 SINGLE FINQA ITEM INFERENCE")
    print("=" * 40)
    
    # Initialize inference
    inference = MistralFinQAInference()
    inference.load_model()
    
    # Load one item from real FinQA data
    try:
        finqa_data = inference.load_finqa_data('data/finqa/test.json')
        finqa_item = finqa_data[0]  # First item
        
        result = inference.process_finqa_item(
            finqa_item=finqa_item,
            show_prompt=True  # Shows the full prompt with context
        )
        
        print(f"\n✅ RESULT:")
        print(f"Response: {result['response']}")
        print(f"Gold Answer: {result['gold_answer']}")
        print(f"Expected Program: {result['expected_program']}")
        
    except FileNotFoundError:
        print("❌ FinQA dataset not found at 'data/finqa/test.json'")
        print("💡 Running with simple question instead...")
        run_single_question()

def run_single_question():
    """Run inference on a single simple question."""
    
    print("🚀 SINGLE QUESTION INFERENCE")
    print("=" * 40)
    
    # Initialize inference
    inference = MistralFinQAInference()
    inference.load_model()
    
    # Single question example
    question = "What is the percentage increase in revenue from $100M to $125M?"
    context = "Company revenue grew from $100M to $125M year-over-year."
    
    result = inference.process_finqa_question(
        question=question,
        context=context,
        show_prompt=True  # Shows the full prompt
    )
    
    print(f"\n✅ RESULT:")
    print(f"Response: {result['response']}")

def run_batch_finqa_data():
    """Run inference on real FinQA dataset."""
    
    print("🚀 REAL FINQA DATASET INFERENCE")
    print("=" * 40)
    
    # Initialize inference
    inference = MistralFinQAInference()
    inference.load_model()
    
    # Run batch inference with real FinQA format
    results = inference.run_batch_inference(
        max_questions=5,  # Process first 5 items
        show_prompts=False,
        use_finqa_format=True  # Use full FinQA format
    )
    
    # Save results
    filename = inference.save_results(results, "real_finqa_results.json")
    print(f"\n📁 Results saved to: {filename}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    for i, result in enumerate(results, 1):
        print(f"Question {i}: {result['question'][:80]}...")
        print(f"  Response: {result['response']}")
        print(f"  Gold: {result['gold_answer']}")
        print(f"  Expected: {result['expected_program']}")
        print()

def run_batch_questions():
    """Run inference on multiple simple questions."""
    
    print("🚀 BATCH INFERENCE")
    print("=" * 40)
    
    # Initialize inference
    inference = MistralFinQAInference()
    inference.load_model()
    
    # Custom questions
    custom_questions = [
        {
            "question": "What is 25% of $2 million?",
            "context": "Calculate 25% of $2,000,000",
            "gold_answer": 500000
        },
        {
            "question": "If costs increased from $50K to $65K, what is the percentage increase?",
            "context": "Operating costs: Previous year $50K, Current year $65K",
            "gold_answer": 30.0
        },
        {
            "question": "What is the average of quarterly revenues: Q1=$8M, Q2=$9.5M, Q3=$7.2M, Q4=$10.1M?",
            "context": "Quarterly data for the fiscal year",
            "gold_answer": 8.7
        }
    ]
    
    # Run inference
    results = inference.run_batch_inference(
        questions=custom_questions,
        show_prompts=False,
        use_finqa_format=False  # Use simple question format
    )
    
    # Save results
    filename = inference.save_results(results, "custom_results.json")
    print(f"\n📁 Results saved to: {filename}")

def run_finqa_sample():
    """Run a small sample of FinQA items for testing."""
    
    print("🚀 FINQA SAMPLE INFERENCE")
    print("=" * 40)
    
    # Initialize inference
    inference = MistralFinQAInference()
    inference.load_model()
    
    try:
        # Load a few items from FinQA
        finqa_data = inference.load_finqa_data('data/finqa/test.json')
        sample_items = finqa_data[:3]  # First 3 items
        
        # Process them
        results = inference.run_batch_inference(
            questions=sample_items,
            max_questions=3,
            show_prompts=False,
            use_finqa_format=True
        )
        
        # Save results
        filename = inference.save_results(results, "finqa_sample_results.json")
        print(f"\n📁 Results saved to: {filename}")
        
        # Show detailed analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        for i, result in enumerate(results, 1):
            print(f"\n--- FinQA Item {i} ---")
            print(f"ID: {result['finqa_id']}")
            print(f"Question: {result['question']}")
            print(f"Model Response: {result['response']}")
            print(f"Expected Program: {result['expected_program']}")
            print(f"Gold Answer: {result['gold_answer']}")
            
            # Check if response contains a function call
            if "FUNCTION:" in result['response']:
                print("✅ Contains function call")
            else:
                print("❌ No function call found")
        
    except FileNotFoundError:
        print("❌ FinQA data file not found. Using sample questions instead.")
        print("💡 To use real FinQA data, place your JSON file at 'data/finqa/test.json'")
        
        # Fall back to sample data
        results = inference.run_batch_inference(
            max_questions=3,
            use_finqa_format=False
        )
        inference.save_results(results, "fallback_results.json")

def test_table_formatting():
    """Test how the system formats FinQA tables."""
    
    print("🚀 TABLE FORMATTING TEST")
    print("=" * 40)
    
    # Initialize inference
    inference = MistralFinQAInference()
    
    try:
        # Load one FinQA item
        finqa_data = inference.load_finqa_data('data/finqa/test.json')
        finqa_item = finqa_data[0]
        
        # Test table formatting
        formatted_context = inference.format_finqa_context(finqa_item)
        
        print("📋 FORMATTED CONTEXT:")
        print("-" * 80)
        print(formatted_context[:1500] + "..." if len(formatted_context) > 1500 else formatted_context)
        print("-" * 80)
        
        print(f"\n📊 CONTEXT STATS:")
        print(f"Total length: {len(formatted_context)} characters")
        print(f"Pre-text items: {len(finqa_item.get('pre_text', []))}")
        print(f"Table rows: {len(finqa_item.get('table', []))}")
        print(f"Post-text items: {len(finqa_item.get('post_text', []))}")
        
    except FileNotFoundError:
        print("❌ FinQA data file not found at 'data/finqa/test.json'")

def main():
    """Main menu for different inference options."""
    
    print("🔥 MISTRAL 7B FINQA INFERENCE OPTIONS")
    print("=" * 50)
    print("1. Single FinQA item (full context)")
    print("2. Single simple question")
    print("3. Batch FinQA dataset (5 items)")
    print("4. Batch simple questions")
    print("5. FinQA sample (3 items with analysis)")
    print("6. Test table formatting")
    print("7. Run all available FinQA items")
    
    choice = input("\nChoose an option (1-7): ").strip()
    
    if choice == "1":
        run_single_finqa_item()
    elif choice == "2":
        run_single_question()
    elif choice == "3":
        run_batch_finqa_data()
    elif choice == "4":
        run_batch_questions()
    elif choice == "5":
        run_finqa_sample()
    elif choice == "6":
        test_table_formatting()
    elif choice == "7":
        # Run all available FinQA items
        inference = MistralFinQAInference()
        inference.load_model()
        results = inference.run_batch_inference(
            max_questions=None,  # Process all items
            use_finqa_format=True
        )
        inference.save_results(results, "all_finqa_results.json")
    else:
        print("❌ Invalid choice. Running FinQA sample...")
        run_finqa_sample()

if __name__ == "__main__":
    main() 