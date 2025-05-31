#!/usr/bin/env python3
"""
Analyze terrible results from successful function executions to identify patterns.
"""

import json

def main():
    with open('evaluation_detailed_results.json', 'r') as f:
        data = json.load(f)

    # Find terrible results - successful execution but very wrong answers
    terrible_results = []
    for entry in data['detailed_results']:
        if entry['execution_success'] and not entry['is_correct'] and not entry['is_close_miss']:
            explanation = entry['explanation']
            if 'rel_diff:' in explanation:
                rel_diff = float(explanation.split('rel_diff: ')[1].split(')')[0])
                if rel_diff > 0.5:  # More than 50% off
                    terrible_results.append({
                        'entry': entry,
                        'rel_diff': rel_diff
                    })

    # Sort by relative difference (worst first)
    terrible_results.sort(key=lambda x: x['rel_diff'], reverse=True)

    print(f'🔍 Found {len(terrible_results)} terrible results (>50% error)')
    print("\n📊 Top 15 Worst Performing Successful Executions:")
    
    for i, result in enumerate(terrible_results[:15]):
        entry = result['entry']
        print(f'\n{i+1}. Q{entry["question_id"]} - Rel Error: {result["rel_diff"]:.1%}')
        print(f'   Question: {entry["question"][:80]}...')
        print(f'   Processed: {entry["processed_response"]}')
        print(f'   Gold: {entry["gold_answer"]}')
        print(f'   Explanation: {entry["explanation"]}')

    # Now let's get the original function calls for these terrible results
    print(f'\n🔧 Analyzing Original Function Calls for Terrible Results:')
    
    # Load the original processed results to get function calls
    with open('mistral_finqa_results_processed.json', 'r') as f:
        processed_data = json.load(f)
    
    # Create a mapping of question_id to original function call
    function_calls = {entry['question_id']: entry['original_response'] for entry in processed_data}
    
    print(f'\n📝 Function Call Analysis:')
    for i, result in enumerate(terrible_results[:10]):
        entry = result['entry']
        qid = entry['question_id']
        original_call = function_calls.get(qid, 'Not found')
        
        print(f'\n{i+1}. Q{qid} (Error: {result["rel_diff"]:.1%})')
        print(f'   Function: {original_call[:100]}...')
        print(f'   Result: {entry["processed_response"]} vs Gold: {entry["gold_answer"]}')
        
        # Analyze potential issues
        issues = []
        if 'percentage_change' in original_call and float(entry["processed_response"]) < -50:
            issues.append("Likely wrong argument order in percentage_change")
        if 'subtract' in original_call and entry["gold_answer"].startswith('-') and float(entry["processed_response"]) > 0:
            issues.append("Wrong subtraction order (should be negative)")
        if 'sum' in original_call and 'percentage' in original_call:
            issues.append("May be using wrong denominator/numerator")
        if float(entry["processed_response"]) > 1000 and '%' in entry["gold_answer"]:
            issues.append("Result too large for percentage - missing division by 100?")
        if abs(float(entry["processed_response"])) > 100 and float(entry["gold_answer"].replace('%', '')) < 100:
            issues.append("Scale mismatch - possibly using raw numbers vs percentages")
            
        if issues:
            print(f'   Likely Issues: {"; ".join(issues)}')
        else:
            print(f'   Issues: Complex logic error')

if __name__ == "__main__":
    main() 