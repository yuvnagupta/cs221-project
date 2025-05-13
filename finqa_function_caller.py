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
            "ratio": self._ratio
        }
        
        # Regular expressions for detecting function calls
        self.function_patterns = {
            "calculator": r"\[CALCULATOR\](.*?)\[/CALCULATOR\]",
            "calendar": r"\[CALENDAR\](.*?)\[/CALENDAR\]",
            "percentage": r"\[PERCENTAGE\](.*?)\[/PERCENTAGE\]",
            "average": r"\[AVERAGE\](.*?)\[/AVERAGE\]",
            "sum": r"\[SUM\](.*?)\[/SUM\]",
            "difference": r"\[DIFFERENCE\](.*?)\[/DIFFERENCE\]",
            "ratio": r"\[RATIO\](.*?)\[/RATIO\]"
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

    def parse_function_calls(self, text: str) -> List[Tuple[str, str, Any]]:
        """Parse function calls from text and return list of (function_name, arguments, result)."""
        results = []
        
        for func_name, pattern in self.function_patterns.items():
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                args = match.group(1).strip()
                try:
                    result = self.functions[func_name](args)
                    results.append((func_name, args, result))
                except Exception as e:
                    print(f"Error executing {func_name} with args '{args}': {str(e)}")
        
        return results

    def format_function_call(self, func_name: str, args: str) -> str:
        """Format a function call in the expected format."""
        return f"[{func_name.upper()}]{args}[/{func_name.upper()}]"

    def process_answer(self, answer: str) -> Tuple[str, List[Any]]:
        """Process an answer containing function calls and return the final answer with results."""
        # Parse all function calls
        function_results = self.parse_function_calls(answer)
        
        # Replace function calls with their results
        processed_answer = answer
        for func_name, args, result in function_results:
            call = self.format_function_call(func_name, args)
            processed_answer = processed_answer.replace(call, str(result))
        
        return processed_answer, function_results

def format_prompt_with_functions() -> str:
    """Return the prompt template that includes function calling instructions."""
    return """Given the following financial text, table, and question, provide a step-by-step solution to find the answer.
You can use the following functions to help with calculations:

1. Calculator: [CALCULATOR]expression[/CALCULATOR]
   Example: [CALCULATOR]100 * 1.1[/CALCULATOR]

2. Calendar: [CALENDAR]date[/CALENDAR]
   Example: [CALENDAR]2024-03-15[/CALENDAR]

3. Percentage: [PERCENTAGE]part of total[/PERCENTAGE]
   Example: [PERCENTAGE]25 of 100[/PERCENTAGE]

4. Average: [AVERAGE]number1, number2, ...[/AVERAGE]
   Example: [AVERAGE]10, 20, 30[/AVERAGE]

5. Sum: [SUM]number1, number2, ...[/SUM]
   Example: [SUM]10, 20, 30[/SUM]

6. Difference: [DIFFERENCE]number1, number2[/DIFFERENCE]
   Example: [DIFFERENCE]100, 50[/DIFFERENCE]

7. Ratio: [RATIO]number1, number2[/RATIO]
   Example: [RATIO]100, 50[/RATIO]

Text:
{context}

Table:
{table_info}

Question: {question}

Let's solve this step by step:"""

def main():
    # Example usage
    caller = FunctionCaller()
    
    # Example answer with function calls
    answer = """
    First, let's calculate the total revenue:
    [SUM]1000000, 2000000, 3000000[/SUM]
    
    Then, let's find the percentage increase:
    [PERCENTAGE]500000 of 6000000[/PERCENTAGE]
    
    Finally, let's calculate the average:
    [AVERAGE]1000000, 2000000, 3000000[/AVERAGE]
    """
    
    # Process the answer
    final_answer, results = caller.process_answer(answer)
    
    print("Original answer:", answer)
    print("\nFunction results:")
    for func_name, args, result in results:
        print(f"{func_name}({args}) = {result}")
    print("\nFinal answer:", final_answer)

if __name__ == "__main__":
    main() 