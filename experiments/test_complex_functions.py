from static_function_parser import StaticFunctionParser
import time

def test_complex_nested_functions():
    """Test the static function parser with complex nested functions and big numbers."""
    
    parser = StaticFunctionParser()
    
    # Complex test cases with big numbers and deep nesting
    complex_test_cases = [
        {
            "name": "Deep Nested Arithmetic",
            "function": "add(multiply(divide(1000000, 4), subtract(500, 100)), percentage(25, 100))",
            "expected": 100025.0,  # (1000000/4) * (500-100) + (25/100)*100 = 250000*400 + 25 = 100000000 + 25
            "description": "Deep nesting with large numbers"
        },
        {
            "name": "Financial Calculation - Revenue Analysis",
            "function": "percentage(add(multiply(1500000, 12), multiply(2300000, 8)), 50000000)",
            "expected": 72.8,  # ((1.5M*12) + (2.3M*8)) / 50M * 100 = (18M + 18.4M) / 50M * 100 = 36.4M/50M*100 = 72.8%
            "description": "Complex revenue percentage calculation"
        },
        {
            "name": "Multi-level Percentage Change",
            "function": "percentage_change(add(1000000, 500000), multiply(add(800000, 700000), 1.25))",
            "expected": 25.0,  # old: 1.5M, new: 1.5M*1.25 = 1.875M, change = (1.875-1.5)/1.5*100 = 25%
            "description": "Percentage change with nested additions and multiplications"
        },
        {
            "name": "Complex Market Share",
            "function": "percentage(add(add(multiply(5000000, 0.15), multiply(3200000, 0.22)), multiply(8100000, 0.18)), 25000000)",
            "expected": 9.516,  # (5M*0.15 + 3.2M*0.22 + 8.1M*0.18) / 25M * 100
            "description": "Market share calculation with multiple product lines"
        },
        {
            "name": "Deeply Nested Division Chain",
            "function": "divide(divide(divide(1000000000, 1000), 500), 2)",
            "expected": 1000.0,  # 1B / 1000 / 500 / 2 = 1M / 500 / 2 = 2000 / 2 = 1000
            "description": "Chain of divisions with billion-scale numbers"
        },
        {
            "name": "Complex Average with Nested Calculations",
            "function": "average(multiply(1200000, 1.05), add(980000, 120000), divide(5500000, 5))",
            "expected": 1100000.0,  # avg(1.26M, 1.1M, 1.1M) = 3.46M/3 ≈ 1.153M
            "description": "Average of complex nested calculations"
        },
        {
            "name": "Ratio with Large Numbers",
            "function": "ratio(multiply(add(750000, 250000), 3), divide(8000000, 2))",
            "expected": "3000000.0:4000000.0",  # (1M * 3) : (8M / 2) = 3M : 4M
            "description": "Ratio calculation with large numbers"
        },
        {
            "name": "Extreme Nesting - 5 Levels Deep",
            "function": "add(multiply(divide(subtract(10000000, 2000000), 4), 3), percentage(add(500000, 300000), 20000000))",
            "expected": 6000004.0,  # ((10M-2M)/4)*3 + (800K/20M)*100 = (8M/4)*3 + 4 = 2M*3 + 4 = 6M + 4
            "description": "Five levels of nesting with mixed operations"
        },
        {
            "name": "Financial Compound Calculation",
            "function": "multiply(add(1, divide(percentage_change(1000000, 1150000), 100)), 5000000)",
            "expected": 5750000.0,  # (1 + (15%/100)) * 5M = 1.15 * 5M = 5.75M
            "description": "Compound interest-style calculation"
        },
        {
            "name": "Max/Min with Complex Nested Values",
            "function": "max(multiply(1500000, 0.8), add(900000, 300000), divide(3600000, 3))",
            "expected": 1200000.0,  # max(1.2M, 1.2M, 1.2M) = 1.2M
            "description": "Maximum of three complex calculations"
        },
        {
            "name": "Percentage of Percentage (Double Nesting)",
            "function": "percentage(percentage(2500000, 50000000), 100)",
            "expected": 5.0,  # (2.5M/50M)*100 = 5%, then 5/100*100 = 5%
            "description": "Percentage of a percentage calculation"
        },
        {
            "name": "Sum of Multiple Complex Operations",
            "function": "sum(multiply(1000000, 1.2), divide(8000000, 4), subtract(3500000, 500000), percentage(10, 100000))",
            "expected": 6200000.1,  # 1.2M + 2M + 3M + 0.1 = 6.2M + 0.1
            "description": "Sum of four different complex operations"
        },
        {
            "name": "Rounding Large Calculations",
            "function": "round(divide(multiply(add(1234567, 2345678), 3), 7), 2)",
            "expected": 1534534.5,  # ((1234567+2345678)*3)/7 rounded to 2 decimals
            "description": "Rounding result of complex calculation"
        }
    ]
    
    print("🧪 TESTING COMPLEX NESTED FUNCTIONS WITH BIG NUMBERS")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(complex_test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"📋 Description: {test_case['description']}")
        print(f"🔧 Function: {test_case['function']}")
        print(f"🎯 Expected: {test_case['expected']}")
        
        start_time = time.time()
        
        try:
            # Test the function
            result = parser.execute_from_text(f"FUNCTION: {test_case['function']}")
            
            execution_time = time.time() - start_time
            
            if result["success"]:
                actual = result["result"]
                print(f"🤖 Actual: {actual}")
                print(f"⏱️ Execution time: {execution_time:.4f}s")
                
                # Check if result matches expected (with tolerance for floating point)
                if isinstance(test_case["expected"], str):
                    # For string results like ratios
                    is_correct = str(actual) == test_case["expected"]
                else:
                    # For numeric results with tolerance
                    tolerance = max(abs(test_case["expected"]) * 0.0001, 0.01)  # 0.01% tolerance or 0.01 minimum
                    is_correct = abs(float(actual) - float(test_case["expected"])) <= tolerance
                
                if is_correct:
                    print("✅ PASSED!")
                    passed += 1
                else:
                    print(f"❌ FAILED! Expected {test_case['expected']}, got {actual}")
                    print(f"   Difference: {abs(float(actual) - float(test_case['expected'])) if not isinstance(test_case['expected'], str) else 'N/A'}")
                    failed += 1
                    
                # Show the parsing breakdown
                print(f"🔤 Tokens: {result['tokens']}")
                
            else:
                print(f"❌ EXECUTION FAILED: {result['error']}")
                failed += 1
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            failed += 1
    
    print(f"\n{'=' * 80}")
    print(f"📊 FINAL TEST RESULTS")
    print(f"{'=' * 80}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The parser handles complex nested functions perfectly!")
    else:
        print(f"⚠️ {failed} tests failed. Review the failures above.")
    
    return passed, failed

def test_edge_cases():
    """Test edge cases and potential failure scenarios."""
    
    parser = StaticFunctionParser()
    
    edge_cases = [
        {
            "name": "Very Large Numbers",
            "function": "add(999999999999, 1)",
            "expected": 1000000000000.0,
            "description": "Handling very large numbers"
        },
        {
            "name": "Many Decimal Places",
            "function": "multiply(1.123456789, 2.987654321)",
            "expected": 3.3548,  # Approximate
            "description": "High precision decimal calculations"
        },
        {
            "name": "Zero Handling",
            "function": "add(multiply(1000000, 0), 42)",
            "expected": 42.0,
            "description": "Multiplication by zero in complex expression"
        },
        {
            "name": "Negative Numbers",
            "function": "subtract(add(1000000, 500000), 2000000)",
            "expected": -500000.0,
            "description": "Handling negative results"
        },
        {
            "name": "Many Arguments",
            "function": "sum(100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000)",
            "expected": 5500000.0,
            "description": "Function with many arguments"
        }
    ]
    
    print(f"\n{'=' * 80}")
    print("🔍 TESTING EDGE CASES")
    print("=" * 80)
    
    edge_passed = 0
    edge_failed = 0
    
    for i, test_case in enumerate(edge_cases, 1):
        print(f"\n🧪 Edge Test {i}: {test_case['name']}")
        print(f"🔧 Function: {test_case['function']}")
        
        try:
            result = parser.execute_from_text(f"FUNCTION: {test_case['function']}")
            
            if result["success"]:
                actual = result["result"]
                
                # More lenient tolerance for edge cases
                tolerance = max(abs(test_case["expected"]) * 0.01, 0.1)
                is_correct = abs(float(actual) - float(test_case["expected"])) <= tolerance
                
                if is_correct:
                    print(f"✅ PASSED! Result: {actual}")
                    edge_passed += 1
                else:
                    print(f"❌ FAILED! Expected ~{test_case['expected']}, got {actual}")
                    edge_failed += 1
            else:
                print(f"❌ FAILED: {result['error']}")
                edge_failed += 1
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            edge_failed += 1
    
    print(f"\n📊 Edge Case Results: {edge_passed} passed, {edge_failed} failed")
    return edge_passed, edge_failed

if __name__ == "__main__":
    # Run comprehensive tests
    main_passed, main_failed = test_complex_nested_functions()
    edge_passed, edge_failed = test_edge_cases()
    
    total_passed = main_passed + edge_passed
    total_failed = main_failed + edge_failed
    
    print(f"\n{'🏆' * 20}")
    print(f"OVERALL RESULTS")
    print(f"{'🏆' * 20}")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Overall Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    
    if total_failed == 0:
        print("🎉 PERFECT SCORE! The static function parser is rock solid!")
    else:
        print(f"⚠️ Room for improvement in {total_failed} cases.") 