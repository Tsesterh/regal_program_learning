#!/bin/bash 

#python program_refactoring/domains/logos/generate_programs.py --in_path logo_data/python/train_200.jsonl --out_path logo_data/images/

from subprocess import Popen


proc = Popen(
        args=[
            'python', "program_refactoring/domains/logos/generate_collection.py",
        ],
    )
proc.wait()