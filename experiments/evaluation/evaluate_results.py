#!/usr/bin/env python3
"""
Evaluate the processed Mistral FinQA results against golden answers.
Accounts for symbols, commas, and rounding errors in comparisons.
"""

import json
import re
import math
from typing import Optional, Tuple


def clean_number_string(text: str) -> Optional[float]:
    """
    Clean a number string by removing symbols, commas, and converting to float.
    Handles percentages, currency symbols, etc.
    """
    if not text or text.strip() == "":
        return None
    
    # Remove common symbols and whitespace
    cleaned = re.sub(r'[,$%\s]+', '', str(text))
    
    # Handle special cases
    if cleaned.lower() in ['yes', 'true']:
        return 1.0
    elif cleaned.lower() in ['no', 'false']:
        return 0.0
    elif cleaned == '':
        return None
    
    # Try to extract number
    # Look for negative numbers, decimals, etc.
    number_match = re.search(r'-?\d*\.?\d+', cleaned)
    if number_match:
        try:
            return float(number_match.group())
        except ValueError:
            return None
    
    return None


def round_to_sig_figs(num: float, sig_figs: int = 3) -> float:
    """
    Round a number to a specified number of significant figures.
    """
    if num == 0:
        return 0.0
    
    # Handle the sign
    sign = 1 if num >= 0 else -1
    num = abs(num)
    
    # Calculate the order of magnitude
    magnitude = math.floor(math.log10(num))
    
    # Round to the specified number of significant figures
    factor = 10 ** (sig_figs - 1 - magnitude)
    rounded = round(num * factor) / factor
    
    return sign * rounded


def are_numbers_close(num1: float, num2: float, relative_tolerance: float = 0.02, absolute_tolerance: float = 0.01) -> bool:
    """
    Compare two numbers with tolerance for rounding errors.
    Uses both relative and absolute tolerance.
    """
    if num1 == num2:
        return True
    
    # Handle zero cases
    if num1 == 0 or num2 == 0:
        return abs(num1 - num2) <= absolute_tolerance
    
    # Relative tolerance check
    relative_diff = abs(num1 - num2) / max(abs(num1), abs(num2))
    if relative_diff <= relative_tolerance:
        return True
    
    # Absolute tolerance check
    return abs(num1 - num2) <= absolute_tolerance


def evaluate_single_result(processed_response: str, gold_answer: str, execution_success: bool) -> Tuple[bool, str]:
    """
    Evaluate a single result against the golden answer.
    Returns (is_correct, explanation)
    """
    if not execution_success:
        return False, "Execution failed"
    
    if processed_response.startswith("ERROR:"):
        return False, "Error in processing"
    
    # Clean both numbers
    processed_num = clean_number_string(processed_response)
    gold_num = clean_number_string(gold_answer)
    
    if processed_num is None:
        return False, f"Could not parse processed result: {processed_response}"
    
    if gold_num is None:
        return False, f"Could not parse gold answer: {gold_answer}"
    
    # Compare numbers with tolerance
    is_close = are_numbers_close(processed_num, gold_num)
    
    if is_close:
        return True, f"Match: {processed_num} ≈ {gold_num}"
    else:
        diff = abs(processed_num - gold_num)
        relative_diff = diff / max(abs(processed_num), abs(gold_num)) if max(abs(processed_num), abs(gold_num)) > 0 else 0
        return False, f"Mismatch: {processed_num} vs {gold_num} (diff: {diff:.4f}, rel_diff: {relative_diff:.4f})"


def main():
    print("📊 Evaluating Mistral FinQA Results...")
    
    # Load processed results
    with open('mistral_finqa_results_processed.json', 'r') as f:
        results = json.load(f)
    
    print(f"📁 Loaded {len(results)} processed results")
    
    # Evaluation metrics
    total_entries = len(results)
    execution_successful = 0
    evaluable_entries = 0
    correct_answers = 0
    close_misses = 0  # Within 10% but outside 5%
    
    # Detailed tracking
    evaluation_details = []
    
    print("\n🔍 Evaluating responses...")
    
    for i, entry in enumerate(results):
        question_id = entry['question_id']
        processed_response = entry['response']
        gold_answer = entry['gold_answer']
        execution_success = entry['execution_success']
        
        if execution_success:
            execution_successful += 1
        
        # Skip entries without gold answers
        if not gold_answer or str(gold_answer).strip() == "":
            continue
        
        evaluable_entries += 1
        
        # Evaluate this entry
        is_correct, explanation = evaluate_single_result(processed_response, gold_answer, execution_success)
        
        # Check if it's a close miss (within 10% but not within 5%)
        is_close_miss = False
        if not is_correct and execution_success:
            processed_num = clean_number_string(processed_response)
            gold_num = clean_number_string(gold_answer)
            if processed_num is not None and gold_num is not None:
                is_close_miss = are_numbers_close(processed_num, gold_num, relative_tolerance=0.10, absolute_tolerance=0.1)
        
        if is_correct:
            correct_answers += 1
        elif is_close_miss:
            close_misses += 1
        
        # Store detailed results
        evaluation_details.append({
            'question_id': question_id,
            'question': entry['question'][:100] + "..." if len(entry['question']) > 100 else entry['question'],
            'processed_response': processed_response,
            'gold_answer': gold_answer,
            'execution_success': execution_success,
            'is_correct': is_correct,
            'is_close_miss': is_close_miss,
            'explanation': explanation
        })
        
        # Print progress every 100 entries
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{total_entries} entries...")
    
    # Calculate metrics
    execution_rate = execution_successful / total_entries * 100
    accuracy_rate = correct_answers / evaluable_entries * 100 if evaluable_entries > 0 else 0
    close_miss_rate = close_misses / evaluable_entries * 100 if evaluable_entries > 0 else 0
    combined_accuracy = (correct_answers + close_misses) / evaluable_entries * 100 if evaluable_entries > 0 else 0
    
    # Calculate accuracy among only successfully executed entries
    executed_evaluable = sum(1 for d in evaluation_details if d['execution_success'] and d['gold_answer'] and str(d['gold_answer']).strip() != "")
    correct_among_executed = sum(1 for d in evaluation_details if d['execution_success'] and d['is_correct'])
    execution_only_accuracy = correct_among_executed / executed_evaluable * 100 if executed_evaluable > 0 else 0
    
    # Print summary
    print(f"\n📈 Evaluation Results:")
    print(f"   Total entries: {total_entries}")
    print(f"   Execution successful: {execution_successful} ({execution_rate:.1f}%)")
    print(f"   Evaluable entries (with gold answers): {evaluable_entries}")
    print(f"   ✅ Correct answers: {correct_answers} ({accuracy_rate:.1f}%)")
    print(f"   🔶 Close misses (within 10%): {close_misses} ({close_miss_rate:.1f}%)")
    print(f"   📊 Combined accuracy (correct + close): {correct_answers + close_misses} ({combined_accuracy:.1f}%)")
    print(f"   🎯 Accuracy among executed only: {correct_among_executed}/{executed_evaluable} ({execution_only_accuracy:.1f}%)")
    
    # Show some examples
    print(f"\n📋 Sample Results:")
    
    # Show first few correct answers
    correct_examples = [d for d in evaluation_details if d['is_correct']][:3]
    print(f"\n✅ Correct Examples:")
    for example in correct_examples:
        print(f"   Q{example['question_id']}: {example['explanation']}")
        print(f"      Question: {example['question']}")
    
    # Show first few incorrect answers
    incorrect_examples = [d for d in evaluation_details if not d['is_correct'] and not d['is_close_miss']][:3]
    print(f"\n❌ Incorrect Examples:")
    for example in incorrect_examples:
        print(f"   Q{example['question_id']}: {example['explanation']}")
        print(f"      Question: {example['question']}")
    
    # Show close misses
    close_miss_examples = [d for d in evaluation_details if d['is_close_miss']][:3]
    if close_miss_examples:
        print(f"\n🔶 Close Miss Examples:")
        for example in close_miss_examples:
            print(f"   Q{example['question_id']}: {example['explanation']}")
            print(f"      Question: {example['question']}")
    
    # Save detailed evaluation results
    output_filename = 'evaluation_detailed_results.json'
    with open(output_filename, 'w') as f:
        json.dump({
            'summary': {
                'total_entries': total_entries,
                'execution_successful': execution_successful,
                'execution_rate': execution_rate,
                'evaluable_entries': evaluable_entries,
                'correct_answers': correct_answers,
                'accuracy_rate': accuracy_rate,
                'close_misses': close_misses,
                'close_miss_rate': close_miss_rate,
                'combined_accuracy': combined_accuracy
            },
            'detailed_results': evaluation_details
        }, f, indent=2)
    
    print(f"\n💾 Detailed evaluation results saved to: {output_filename}")


if __name__ == "__main__":
    main() 