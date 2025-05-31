#!/usr/bin/env python3
"""
Static Function Parser for Nested Mathematical Expressions

A Python library for parsing and executing nested mathematical function calls 
from natural language model outputs. Designed specifically for financial analysis
and mathematical computations where models need to perform complex calculations.

Example:
    parser = StaticFunctionParser()
    result = parser.execute_from_text("FUNCTION: add(multiply(2, 3), divide(8, 2))")
    print(result['result'])  # Output: 10.0

Author: CS221 Project Team
License: MIT
"""

import re
import json
from typing import Any, Dict, List, Union, Tuple
from collections import deque


class StaticFunctionParser:
    """
    A static function parser that executes nested function calls from model output.
    
    This class uses a recursive descent parser to handle nested mathematical expressions
    and execute them using a predefined set of mathematical functions. It's designed
    to prevent hallucination by executing only the first valid function call found
    in model output and returning the numerical result immediately.
    
    Supported Functions:
        - Basic arithmetic: add, subtract, multiply, divide
        - Statistical: average, sum, max, min
        - Financial: percentage, percentage_change, ratio
        - Utility: round
    
    Attributes:
        functions (dict): Dictionary mapping function names to their implementations
    """
    
    def __init__(self):
        """Initialize the parser with available mathematical functions."""
        self.functions = {
            "add": self._add,
            "subtract": self._subtract,
            "multiply": self._multiply,
            "divide": self._divide,
            "percentage": self._percentage,
            "percentage_change": self._percentage_change,
            "ratio": self._ratio,
            "average": self._average,
            "sum": self._sum,
            "max": self._max,
            "min": self._min,
            "round": self._round,
        }
    
    # Mathematical function implementations
    def _add(self, *args) -> float:
        """Add multiple numbers."""
        return sum(float(x) for x in args)
    
    def _subtract(self, a, b) -> float:
        """Subtract b from a."""
        return float(a) - float(b)
    
    def _multiply(self, *args) -> float:
        """Multiply multiple numbers."""
        result = 1
        for x in args:
            result *= float(x)
        return result
    
    def _divide(self, a, b) -> float:
        """Divide a by b."""
        if float(b) == 0:
            raise ValueError("Division by zero")
        return float(a) / float(b)
    
    def _percentage(self, part, whole) -> float:
        """Calculate percentage: (part/whole) * 100."""
        if float(whole) == 0:
            raise ValueError("Division by zero in percentage calculation")
        return (float(part) / float(whole)) * 100
    
    def _percentage_change(self, old_value, new_value) -> float:
        """Calculate percentage change: ((new-old)/old) * 100."""
        if float(old_value) == 0:
            raise ValueError("Division by zero in percentage change calculation")
        return ((float(new_value) - float(old_value)) / float(old_value)) * 100
    
    def _ratio(self, a, b) -> str:
        """Calculate ratio a:b."""
        return f"{float(a)}:{float(b)}"
    
    def _average(self, *args) -> float:
        """Calculate average of numbers."""
        numbers = [float(x) for x in args]
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        return sum(numbers) / len(numbers)
    
    def _sum(self, *args) -> float:
        """Sum multiple numbers."""
        return sum(float(x) for x in args)
    
    def _max(self, *args) -> float:
        """Find maximum of numbers."""
        if not args:
            raise ValueError("Cannot find max of empty list")
        return max(float(x) for x in args)
    
    def _min(self, *args) -> float:
        """Find minimum of numbers."""
        if not args:
            raise ValueError("Cannot find min of empty list")
        return min(float(x) for x in args)
    
    def _round(self, number, decimals=2) -> float:
        """Round number to specified decimal places."""
        return round(float(number), int(decimals))
    
    def tokenize_expression(self, expression: str) -> List[str]:
        """
        Tokenize a function expression into tokens.
        
        Args:
            expression: Function expression string (e.g., "add(5, multiply(2, 3))")
            
        Returns:
            List of tokens: function names, numbers, parentheses, commas
        """
        expression = expression.strip()
        
        # Pattern to match function names, numbers, parentheses, commas
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[(),])'
        tokens = re.findall(pattern, expression)
        
        # Clean up tokens
        tokens = [token.strip() for token in tokens if token.strip()]
        return tokens
    
    def parse_function_call(self, tokens: List[str], start_idx: int = 0) -> Tuple[float, int]:
        """
        Parse a function call from tokens starting at start_idx.
        
        Uses recursive descent parsing to handle nested function calls.
        
        Args:
            tokens: List of tokenized expression elements
            start_idx: Starting index in the tokens list
            
        Returns:
            Tuple of (result, next_index)
            
        Raises:
            ValueError: If the expression is malformed or contains unknown functions
        """
        if start_idx >= len(tokens):
            raise ValueError("Unexpected end of expression")
        
        func_name = tokens[start_idx]
        
        if func_name not in self.functions:
            # If it's a number, return it
            try:
                return float(func_name), start_idx + 1
            except ValueError:
                raise ValueError(f"Unknown function or invalid number: {func_name}")
        
        # Expect opening parenthesis
        if start_idx + 1 >= len(tokens) or tokens[start_idx + 1] != '(':
            raise ValueError(f"Expected '(' after function name {func_name}")
        
        # Parse arguments
        args = []
        idx = start_idx + 2  # Skip function name and opening parenthesis
        
        while idx < len(tokens) and tokens[idx] != ')':
            if tokens[idx] == ',':
                idx += 1
                continue
            
            # Check if this is a nested function call
            if idx < len(tokens) - 1 and tokens[idx + 1] == '(':
                # Recursive call for nested function
                result, idx = self.parse_function_call(tokens, idx)
                args.append(result)
            else:
                # Simple argument (number)
                try:
                    args.append(float(tokens[idx]))
                    idx += 1
                except ValueError:
                    raise ValueError(f"Invalid argument: {tokens[idx]}")
        
        if idx >= len(tokens):
            raise ValueError("Missing closing parenthesis")
        
        # Execute the function
        try:
            result = self.functions[func_name](*args)
            return result, idx + 1  # Skip closing parenthesis
        except Exception as e:
            raise ValueError(f"Error executing {func_name}: {str(e)}")
    
    def extract_function_call(self, text: str) -> str:
        """
        Extract function call from model output text.
        
        Searches for common patterns that indicate function calls in model output,
        such as "FUNCTION: ..." or standalone function expressions.
        
        Args:
            text: Raw text from model output
            
        Returns:
            Extracted function call string, or empty string if none found
        """
        # Try different patterns to find function calls
        patterns = [
            r'FUNCTION:\s*(.+?)(?:\n|$)',
            r'CALL:\s*(.+?)(?:\n|$)',
            r'EXECUTE:\s*(.+?)(?:\n|$)',
            r'CALCULATE:\s*(.+?)(?:\n|$)',
            r'(?:^|\n)([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, look for any function-like expression
        func_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\))'
        matches = re.findall(func_pattern, text)
        if matches:
            # Return the last (most complete) match
            return matches[-1].strip()
        
        return ""
    
    def execute_from_text(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Execute function calls found in text and return results.
        
        This is the main entry point for parsing model output and executing
        the mathematical functions found within.
        
        Args:
            text: Raw text containing function calls
            verbose: Whether to print debug information
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating if execution succeeded
                - result: The numerical result (if successful)
                - function_call: The extracted function call
                - error: Error message (if failed)
        """
        if verbose:
            print(f"🔍 Parsing text: {text[:200]}...")
        
        # Extract function call
        function_call = self.extract_function_call(text)
        
        if not function_call:
            return {
                "success": False,
                "error": "No function call found in text",
                "text": text
            }
        
        if verbose:
            print(f"📞 Found function call: {function_call}")
        
        try:
            # Tokenize the expression
            tokens = self.tokenize_expression(function_call)
            if verbose:
                print(f"🔤 Tokens: {tokens}")
            
            # Parse and execute
            result, _ = self.parse_function_call(tokens, 0)
            
            return {
                "success": True,
                "function_call": function_call,
                "result": result,
                "tokens": tokens if verbose else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "function_call": function_call,
                "text": text
            }
    
    def get_available_functions(self) -> List[str]:
        """Return list of available function names."""
        return list(self.functions.keys())
    
    def create_prompt_template(self, question: str) -> str:
        """
        Create a prompt template that encourages models to output
        a single nested function call.
        
        Args:
            question: The question to be answered
            
        Returns:
            Formatted prompt string
        """
        available_functions = ", ".join(self.functions.keys())
        
        template = f"""You are a mathematical assistant. Given a question, you must respond with EXACTLY ONE function call that solves the problem.

Available functions: {available_functions}

Rules:
1. Use nested function calls when needed (e.g., add(multiply(2, 3), 4))
2. Always output your final answer as: FUNCTION: your_function_call_here
3. Do not explain or add extra text after the function call
4. Use only the available functions listed above

Question: {question}

FUNCTION: """
        
        return template


def main():
    """Example usage and basic testing of the StaticFunctionParser."""
    parser = StaticFunctionParser()
    
    # Test cases demonstrating various function call patterns
    test_cases = [
        "add(5, 3)",
        "multiply(add(2, 3), 4)",
        "divide(subtract(100, 20), 4)",
        "percentage(multiply(3, 4), 60)",
        "add(multiply(2, 3), divide(8, 2))",
        "average(10, 20, 30, 40)",
        "percentage_change(100, 120)",
    ]
    
    print("🧪 Testing Static Function Parser")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\n📝 Testing: {test}")
        result = parser.execute_from_text(f"FUNCTION: {test}")
        
        if result["success"]:
            print(f"✅ Result: {result['result']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Test with realistic model output
    print("\n" + "=" * 50)
    print("🤖 Testing with model-like output")
    
    model_output = """
    To solve this problem, I need to calculate the total revenue.
    First, I'll multiply the price by quantity, then add the bonus.
    
    FUNCTION: add(multiply(25, 4), 100)
    
    This gives us the final answer.
    """
    
    result = parser.execute_from_text(model_output)
    if result["success"]:
        print(f"✅ Model output result: {result['result']}")
    else:
        print(f"❌ Model output error: {result['error']}")


if __name__ == "__main__":
    main() 