# Static Function Parser for LLMs

A robust Python library for parsing and executing nested mathematical function calls from Large Language Model (LLM) outputs. Designed specifically for financial analysis and mathematical computations where models need to perform complex calculations without hallucination.

## 🎯 Problem Solved

Large Language Models often hallucinate when performing mathematical calculations, especially after function calls. This library prevents hallucination by:

1. **Static Parsing**: Executes only the first valid function call found
2. **Immediate Results**: Returns numerical results immediately without allowing continued generation
3. **Nested Support**: Handles complex nested function expressions
4. **Error Prevention**: Robust error handling for malformed expressions

## 🚀 Features

- **Nested Function Calls**: Support for complex expressions like `add(multiply(2, 3), divide(8, 2))`
- **Mathematical Functions**: Comprehensive set of mathematical and statistical functions
- **Pattern Recognition**: Intelligent extraction of function calls from natural language
- **Type Safety**: Full type hints and comprehensive error handling
- **Zero Dependencies**: Only uses Python standard library

## 📦 Installation

```bash
git clone https://github.com/your-username/static-function-parser.git
cd static-function-parser
```

No additional dependencies required - uses only Python standard library.

## 🔧 Quick Start

```python
from static_function_parser import StaticFunctionParser

# Initialize parser
parser = StaticFunctionParser()

# Parse model output containing function calls
model_output = """
To calculate the total revenue, I need to multiply price by quantity and add the bonus.
FUNCTION: add(multiply(25, 4), 100)
This gives us the final answer.
"""

# Execute and get result
result = parser.execute_from_text(model_output)
print(result['result'])  # Output: 200.0
```

## 📊 Supported Functions

### Basic Arithmetic
- `add(a, b, ...)` - Addition of multiple numbers
- `subtract(a, b)` - Subtraction (a - b)
- `multiply(a, b, ...)` - Multiplication of multiple numbers
- `divide(a, b)` - Division (a / b)

### Statistical Functions
- `average(a, b, ...)` - Arithmetic mean
- `sum(a, b, ...)` - Sum of multiple numbers
- `max(a, b, ...)` - Maximum value
- `min(a, b, ...)` - Minimum value

### Financial Functions
- `percentage(part, whole)` - Calculate percentage
- `percentage_change(old, new)` - Calculate percentage change
- `ratio(a, b)` - Calculate ratio a:b

### Utility Functions
- `round(number, decimals)` - Round to specified decimal places

## 💡 Usage Examples

### Simple Function Call
```python
parser = StaticFunctionParser()
result = parser.execute_from_text("FUNCTION: add(10, 20)")
print(result['result'])  # 30.0
```

### Nested Function Call
```python
result = parser.execute_from_text("FUNCTION: percentage(multiply(3, 4), 60)")
print(result['result'])  # 20.0 (12 is 20% of 60)
```

### Complex Financial Calculation
```python
# Calculate percentage change in revenue
text = "FUNCTION: percentage_change(add(100, 50), multiply(200, 0.8))"
result = parser.execute_from_text(text)
print(result['result'])  # -6.25 (percentage change from 150 to 160)
```

### Extracting from Natural Language
```python
model_response = """
Based on the financial data, I need to calculate the growth rate.
The revenue increased from $1M to $1.2M.
FUNCTION: percentage_change(1000000, 1200000)
Therefore, the growth rate is 20%.
"""

result = parser.execute_from_text(model_response)
print(result['result'])  # 20.0
```

## 🔍 API Reference

### `StaticFunctionParser`

#### `execute_from_text(text: str, verbose: bool = False) -> Dict[str, Any]`

Main method to parse and execute function calls from text.

**Parameters:**
- `text`: Input text containing function calls
- `verbose`: Enable debug output (default: False)

**Returns:**
Dictionary with:
- `success`: Boolean indicating success/failure
- `result`: Numerical result (if successful)
- `function_call`: Extracted function call string
- `error`: Error message (if failed)

#### `get_available_functions() -> List[str]`

Returns list of all available function names.

#### `create_prompt_template(question: str) -> str`

Creates a prompt template for LLMs to generate proper function calls.

## 🧪 Testing

Run the built-in tests:

```bash
python static_function_parser.py
```

This will run comprehensive tests covering:
- Basic arithmetic operations
- Nested function calls
- Error handling
- Natural language extraction

## 🔧 Development

### Project Structure
```
static-function-parser/
├── static_function_parser.py    # Main parser implementation
├── README.md                   # This file
├── requirements.txt            # Dependencies (none currently)
└── examples/                   # Usage examples
```

### Adding New Functions

To add a new mathematical function:

1. Add the function to the `functions` dictionary in `__init__`
2. Implement the function with prefix `_function_name`
3. Add appropriate error handling
4. Update documentation

Example:
```python
def __init__(self):
    self.functions = {
        # ... existing functions ...
        "power": self._power,
    }

def _power(self, base, exponent) -> float:
    """Calculate base raised to the power of exponent."""
    return float(base) ** float(exponent)
```

## 🎯 Use Cases

### Financial Analysis
- Revenue calculations with multiple components
- Percentage change analysis
- Ratio calculations for financial metrics
- Statistical analysis of financial data

### Educational Tools
- Mathematical expression evaluation
- Step-by-step calculation verification
- Safe execution of student-generated expressions

### LLM Integration
- Preventing mathematical hallucination
- Standardizing numerical outputs
- Enabling reliable quantitative reasoning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Developed for CS221 Project on LLM Mathematical Reasoning
- Inspired by the need for reliable mathematical computation in AI systems
- Built to solve hallucination problems in financial analysis applications

## 📚 Related Work

This library addresses the hallucination problem identified in:
- LLM mathematical reasoning research
- Financial AI applications
- Tool-augmented language models

For academic use, please cite:
```bibtex
@misc{static-function-parser,
  title={Static Function Parser for LLM Mathematical Reasoning},
  author={CS221 Project Team},
  year={2024},
  url={https://github.com/your-username/static-function-parser}
}
``` 