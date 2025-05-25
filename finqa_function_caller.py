import re
import json
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

class FunctionCaller:
    """Handles function calls for FinQA, including calculator and calendar operations."""
    
    def __init__(self):
        self.functions = {
            "calculator": self._calculator,
            "calendar": self._calendar,
            "percentage": self._percentage,
            "average": self._average,
            "sum": self._sum,
            "difference": self._difference,
            "ratio": self._ratio,
            "final_answer": lambda x: x  # Now FINAL_ANSWER is recognized
        }
        
        # Regular expressions for detecting function calls
        self.function_patterns = {
            "calculator": r"\[CALCULATOR\](.*?)\[/CALCULATOR\]",
            "calendar": r"\[CALENDAR\](.*?)\[/CALENDAR\]",
            "percentage": r"\[PERCENTAGE\](.*?)\[/PERCENTAGE\]",
            "average": r"\[AVERAGE\](.*?)\[/AVERAGE\]",
            "sum": r"\[SUM\](.*?)\[/SUM\]",
            "difference": r"\[DIFFERENCE\](.*?)\[/DIFFERENCE\]",
            "ratio": r"\[RATIO\](.*?)\[/RATIO\]",
            "final_answer": r"\[FINAL_ANSWER\](.*?)\[/FINAL_ANSWER\]"
        }

    def _calculator(self, expression: str) -> float:
        """Evaluate a mathematical expression safely."""
        # Remove any whitespace
        expression = expression.strip()
        
        # Replace common financial terms with their mathematical equivalents
        expression = expression.replace("percent", "/100")
        expression = expression.replace("%", "/100")
        
        # Only allow basic arithmetic operations and numbers
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError(f"Invalid characters in expression: {expression}")
        
        try:
            # Use eval with a restricted environment
            result = eval(expression, {"__builtins__": {}}, {"math": math})
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")

    def _calendar(self, date_str: str) -> str:
        """Handle date calculations and formatting."""
        try:
            # Try different date formats
            formats = [
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%Y/%m/%d",
                "%B %d, %Y",
                "%d %B %Y"
            ]
            
            date = None
            for fmt in formats:
                try:
                    date = datetime.strptime(date_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if date is None:
                raise ValueError(f"Could not parse date: {date_str}")
            
            return date.strftime("%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Error processing date '{date_str}': {str(e)}")

    def _percentage(self, values: str) -> float:
        """Calculate percentage of a value."""
        try:
            # Parse values like "25 of 100" or "25/100"
            if " of " in values:
                part, total = values.split(" of ")
            elif "/" in values:
                part, total = values.split("/")
            elif "|" in values:
                part, total = values.split("|")
            else:
                raise ValueError("Invalid percentage format")
            
            part = float(part.strip())
            total = float(total.strip())
            
            if total == 0:
                raise ValueError("Cannot calculate percentage with zero total")
            
            return (part / total) * 100
        except Exception as e:
            raise ValueError(f"Error calculating percentage: {str(e)}")

    def _average(self, values: str) -> float:
        """Calculate average of a list of numbers."""
        try:
            # Parse comma-separated values
            numbers = [float(x.strip()) for x in values.split(",")]
            if not numbers:
                raise ValueError("No numbers provided")
            return sum(numbers) / len(numbers)
        except Exception as e:
            raise ValueError(f"Error calculating average: {str(e)}")

    def _sum(self, values: str) -> float:
        """Calculate sum of a list of numbers."""
        try:
            # Parse comma-separated values
            numbers = [float(x.strip()) for x in values.split(",")]
            if not numbers:
                raise ValueError("No numbers provided")
            return sum(numbers)
        except Exception as e:
            raise ValueError(f"Error calculating sum: {str(e)}")

    def _difference(self, values: str) -> float:
        """Calculate difference between two numbers."""
        try:
            # Parse comma-separated values
            numbers = [float(x.strip()) for x in values.split(",")]
            if len(numbers) != 2:
                raise ValueError("Exactly two numbers required")
            return abs(numbers[0] - numbers[1])
        except Exception as e:
            raise ValueError(f"Error calculating difference: {str(e)}")

    def _ratio(self, values: str) -> float:
        """Calculate ratio between two numbers."""
        try:
            # Parse comma-separated values
            numbers = [float(x.strip()) for x in values.split(",")]
            if len(numbers) != 2:
                raise ValueError("Exactly two numbers required")
            if numbers[1] == 0:
                raise ValueError("Cannot calculate ratio with zero denominator")
            return numbers[0] / numbers[1]
        except Exception as e:
            raise ValueError(f"Error calculating ratio: {str(e)}")

    def process_answer(self, answer: str) -> Tuple[str, List[Tuple[str, str, Any]]]:
        """Process an answer containing function calls and return the final answer with results."""
        # Extract all function calls
        pattern = r"\[(\w+)\](.*?)\[/\1\]"
        matches = re.finditer(pattern, answer, re.DOTALL)
        
        results = []
        final_value = None
        
        # Process each function call in sequence
        for match in matches:
            func_name = match.group(1).lower()
            args = match.group(2)
            
            if func_name in self.functions:
                try:
                    result = self.functions[func_name](args)
                    results.append((func_name, args, result))
                    final_value = result  # Keep track of the last result
                except Exception as e:
                    print(f"Error in {func_name} with args '{args}': {str(e)}")
                    results.append((func_name, args, f"ERROR: {str(e)}"))
        
        # If we have a final value, wrap it in FINAL_ANSWER tags
        if final_value is not None:
            return f"[FINAL_ANSWER]{final_value}[/FINAL_ANSWER]", results
        return answer, results

    def format_function_call(self, func_name: str, args: str) -> str:
        """Format a function call in the expected format."""
        return f"[{func_name.upper()}]{args}[/{func_name.upper()}]"

def format_prompt_with_functions() -> str:
    """Return the prompt template that includes function calling instructions."""
    return """You are a financial analysis assistant. Your task is to analyze the given financial text and table to answer the question.

IMPORTANT INSTRUCTIONS:
1. First, carefully analyze the table structure and data types provided
2. Use the appropriate functions for calculations
3. Show your work step by step
4. Always include the final answer in the exact format specified

AVAILABLE FUNCTIONS:
1. Calculator: [CALCULATOR]expression[/CALCULATOR]
   Example: [CALCULATOR]100 * 1.1[/CALCULATOR]
   Note: Only use basic arithmetic operations (+, -, *, /)

2. Calendar: [CALENDAR]date[/CALENDAR]
   Example: [CALENDAR]2024-03-15[/CALENDAR]
   Note: Use YYYY-MM-DD format

3. Percentage: [PERCENTAGE]part of total[/PERCENTAGE] or [PERCENTAGE]part|total[/PERCENTAGE]
   Example: [PERCENTAGE]25 of 100[/PERCENTAGE] or [PERCENTAGE]25|100[/PERCENTAGE]
   Note: Returns percentage value (e.g., 25.0 for 25%)

4. Percentage Change: [PERCENTAGE_CHANGE]old_value|new_value[/PERCENTAGE_CHANGE]
   Example: [PERCENTAGE_CHANGE]100|120[/PERCENTAGE_CHANGE]
   Note: Returns percentage change (e.g., 20.0 for 20% increase)

5. Average: [AVERAGE]number1, number2, ...[/AVERAGE]
   Example: [AVERAGE]10, 20, 30[/AVERAGE]
   Note: Returns arithmetic mean

6. Sum: [SUM]number1, number2, ...[/SUM]
   Example: [SUM]10, 20, 30[/SUM]
   Note: Returns total of all numbers

7. Difference: [DIFFERENCE]value1|value2[/DIFFERENCE]
   Example: [DIFFERENCE]100|50[/DIFFERENCE]
   Note: Returns absolute difference

8. Ratio: [RATIO]number1, number2[/RATIO]
   Example: [RATIO]100, 50[/RATIO]
   Note: Returns first number divided by second

Text:
{context}

Table:
{table_info}

Question: {question}

REQUIRED ANSWER FORMAT:
1. First, explain what data you need from the table
2. Then, show your calculations using the functions above
3. Finally, provide your answer in this exact format:
[FINAL_ANSWER]your_numeric_answer[/FINAL_ANSWER]

Examples of final answers:
- For percentages: [FINAL_ANSWER]25.5[/FINAL_ANSWER]
- For currency: [FINAL_ANSWER]1234.56[/FINAL_ANSWER]
- For ratios: [FINAL_ANSWER]0.75[/FINAL_ANSWER]

IMPORTANT:
- Do not include any units or symbols in the final answer
- Round to 2 decimal places unless specified otherwise
- If the answer cannot be calculated, use [FINAL_ANSWER]N/A[/FINAL_ANSWER]
- Do not include any text after the final answer tag
- Do not include follow-up exercises or additional questions
- Use the function tags exactly as shown in the examples
- Show your work using the function tags, not in natural language"""

def main():
    # Example usage
    caller = FunctionCaller()
    
    # Test case 1: Sequential function calls
    test1 = """
    Step 1: Calculate the new value
    [CALCULATOR]1000+200[/CALCULATOR]
    
    Step 2: Calculate percentage change
    [PERCENTAGE_CHANGE]1000|1200[/PERCENTAGE_CHANGE]
    
    Step 3: Final answer
    [FINAL_ANSWER]20.0[/FINAL_ANSWER]
    """
    print("\nTest 1 - Sequential function calls:")
    print("Input:", test1)
    result1, steps1 = caller.process_answer(test1)
    print("Output:", result1)
    print("Steps:", steps1)
    
    # Test case 2: Multiple calculations
    test2 = """
    Step 1: Calculate sum
    [SUM]100,200[/SUM]
    
    Step 2: Calculate total
    [CALCULATOR]500+100[/CALCULATOR]
    
    Step 3: Calculate percentage
    [PERCENTAGE]300|600[/PERCENTAGE]
    
    Step 4: Final answer
    [FINAL_ANSWER]50.0[/FINAL_ANSWER]
    """
    print("\nTest 2 - Multiple calculations:")
    print("Input:", test2)
    result2, steps2 = caller.process_answer(test2)
    print("Output:", result2)
    print("Steps:", steps2)
    
    # Test case 3: Complex calculation with multiple steps
    test3 = """
    Step 1: Calculate total revenue
    [SUM]1000000, 2000000, 3000000[/SUM]
    
    Step 2: Calculate new value
    [CALCULATOR]1000000+2000000[/CALCULATOR]
    
    Step 3: Calculate percentage change
    [PERCENTAGE_CHANGE]1000000|3000000[/PERCENTAGE_CHANGE]
    
    Step 4: Final answer
    [FINAL_ANSWER]200.0[/FINAL_ANSWER]
    """
    print("\nTest 3 - Complex calculation:")
    print("Input:", test3)
    result3, steps3 = caller.process_answer(test3)
    print("Output:", result3)
    print("Steps:", steps3)

if __name__ == "__main__":
    main() 