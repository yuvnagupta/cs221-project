#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for evaluate results
Usage:
python evaluate.py predicted_file_name test_file_name
"""
import time
import os
import sys
import shutil
import io
import re
import json
import copy
import random
import collections
import math
import numpy as np
from tqdm import tqdm
from sympy import simplify


all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max", \
"table_min", "table_sum", "table_average"]

def str_to_num(text):
    
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num

def process_row(row_in):
    
    row_out = []
    invalid_flag = 0
    
    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()
        
        num = str_to_num(num)
        
        if num == "n/a":
            invalid_flag = 1
            break
        
        row_out.append(num)
        
    if invalid_flag:
        return "n/a"
    
    return row_out


def eval_program(program, table):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"
    
    try:
        program = program[:-1] # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"


        program = "|".join(program)
        steps = program.split(")")[:-1]
        
        
        res_dict = {}
        
        # print(program)
        
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            
            # print(args)
            # print(op)
            
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            
            if op == "add" or op == "subtract" or op == "multiply" or op == "divide" or op == "exp" or op == "greater":
                
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    # print(arg1)
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break
                
                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break
                
                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"

                    
                # print("ind: ", ind)
                # print(this_res)
                res_dict[ind] = this_res


            elif "table" in op:
                table_dict = {}
                for row in table:
                    table_dict[row[0]] = row[1:]
                    
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    
                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)
                    
                if num_row == "n/a":
                    invalid_flag = 1
                    break
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row)
                    
                # this_res = round(this_res, 5)

                res_dict[ind] = this_res

            # print(this_res)

        if this_res != "yes" and this_res != "no" and this_res != "n/a":
            # print(this_res)
            this_res = round(this_res, 5)

    except:
        invalid_flag = 1
        

    return invalid_flag, this_res


def equal_program(program1, program2):
    '''
    symbolic program if equal
    program1: gold
    program2: pred
    '''
    
    sym_map = {}
    
    program1 = program1[:-1] # remove EOF
    program1 = "|".join(program1)
    steps = program1.split(")")[:-1]
    
    invalid_flag = 0
    sym_ind = 0
    step_dict_1 = {}
    
    # symbolic map
    for ind, step in enumerate(steps):
        
        step = step.strip()

        assert len(step.split("(")) <= 2

        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()
        
        step_dict_1[ind] = step

        if "table" in op:
            if step not in sym_map:
                sym_map[step] = "a" + str(sym_ind)
                sym_ind += 1
                
        else:
            if "#" not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = "a" + str(sym_ind)
                    sym_ind += 1
                    
            if "#" not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = "a" + str(sym_ind)
                    sym_ind += 1


    # check program 2
    step_dict_2 = {}
    try:
        program2 = program2[:-1] # remove EOF
        # check structure
        for ind, token in enumerate(program2):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    print("structure error")
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    print("structure error")
                    return False

        program2 = "|".join(program2)
        steps = program2.split(")")[:-1]
        
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            
            # print(args)
            # print(op)
            
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            
            step_dict_2[ind] = step

            if "table" in op:
                if step not in sym_map:
                    return False
                    
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    if int(arg1.strip("#")) >= ind:
                        return False
                        
                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip("#")) >= ind:
                        return False
    except:
        return False

    def symbol_recur(step, step_dict):
        
        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()
        
        # print(op)
        # print(arg1)
        # print(arg2)
        
        if "table" in op:
            # as var
            return sym_map[step]
        
        if "#" in arg1:
            arg1_ind = int(arg1.replace("#", ""))
            arg1_part = symbol_recur(step_dict[arg1_ind], step_dict)
        else:
            arg1_part = sym_map[arg1]
            
            
        if "#" in arg2:
            arg2_ind = int(arg2.replace("#", ""))
            arg2_part = symbol_recur(step_dict[arg2_ind], step_dict)
        else:
            arg2_part = sym_map[arg2]
            
        if op == "add":
            return "( " + arg1_part + " + " + arg2_part + " )"
        elif op == "subtract":
            return "( " + arg1_part + " - " + arg2_part + " )"
        elif op == "multiply":
            return "( " + arg1_part + " * " + arg2_part + " )"
        elif op == "divide":
            return "( " + arg1_part + " / " + arg2_part + " )"
        elif op == "exp":
            return "( " + arg1_part + " ** " + arg2_part + " )"
        elif op == "greater":
            return "( " + arg1_part + " > " + arg2_part + " )"


    # # derive symbolic program 1
    # print(program1)
    steps = program1.split(")")[:-1]
    # print(steps)
    # print(steps)
    # print(sym_map)
    sym_prog1 = symbol_recur(steps[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)
    # print("########")
    # print(sym_prog1)
    
    try:
        # derive symbolic program 2
        steps = program2.split(")")[:-1]
        sym_prog2 = symbol_recur(steps[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
        # print(sym_prog2)
    except:
        return False

    return sym_prog1 == sym_prog2


def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


def evaluate_result(json_in, json_ori):
    """
    Simple Exact Match Evaluation
    """
    correct = 0

    with open(json_in) as f_in:
        data = json.load(f_in)

    with open(json_ori) as f_in:
        data_ori = json.load(f_in)

    data_dict = {ex["id"]: ex for ex in data_ori}

    for each_data in data:
        each_id = each_data["id"]
        each_ori_data = data_dict[each_id]

        pred = str(each_data["predicted"]).strip().lower()
        gold = str(each_ori_data["qa"]["exe_ans"]).strip().lower()

        if pred == gold:
            correct += 1

    acc = correct / len(data)
    print("All:", len(data))
    print("Exact Match Accuracy:", acc)
    return acc


if __name__ == '__main__':


    json_in = sys.argv[1]
    json_ori = sys.argv[2]

    evaluate_result(json_in, json_ori)
    
  