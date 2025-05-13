# FinQA with Function Calling

This project implements various approaches to solve the FinQA dataset using different models and techniques, including function calling for improved numerical reasoning.

## Project Structure

- `finqa_function_caller.py`: Implementation of the function caller for numerical operations
- `finqa_inference.py`: Inference using Phi-2 model with function calling
- `finqa_gpt35_baseline.py`: Baseline implementation using GPT-3.5 without function calling
- `finqa_gpt35_function.py`: Implementation using GPT-3.5 with function calling
- `phi2_baseline.py`: Baseline implementation using Phi-2 model
- `requirements.txt`: Project dependencies

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cs221-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key (for GPT-3.5 experiments):
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Running Phi-2 Baseline
```bash
python phi2_baseline.py
```

### Running Phi-2 with Function Calling
```bash
python finqa_inference.py
```

### Running GPT-3.5 Baseline
```bash
python finqa_gpt35_baseline.py
```

### Running GPT-3.5 with Function Calling
```bash
python finqa_gpt35_function.py
```

## Function Calling

The function caller supports the following operations:
1. Calculator: Basic arithmetic operations
2. Calendar: Date formatting and calculations
3. Percentage: Percentage calculations
4. Average: Computing averages
5. Sum: Summing numbers
6. Difference: Computing differences
7. Ratio: Computing ratios

## Evaluation

All implementations use the official FinQA evaluation script. Results are saved in the `outputs/` directory.

## Results

Results for each implementation are saved in the following files:
- Phi-2 Baseline: `outputs/finqa_predictions.json`
- Phi-2 with Functions: `outputs/finqa_function_predictions.json`
- GPT-3.5 Baseline: `outputs/finqa_gpt35_baseline_predictions.json`
- GPT-3.5 with Functions: `outputs/finqa_gpt35_function_predictions.json`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 