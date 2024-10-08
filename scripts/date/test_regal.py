from subprocess import Popen
import datetime
import os
import argparse

#read args from command line, "type", "seed", "max_budget", "budget_split"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="use_abstraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_budget", type=int, default=10)
    parser.add_argument("--budget_split", type=float, default=0.5)
    args = parser.parse_args()

    exp_type = args.exp_type
    seed = args.seed
    max_budget = args.max_budget
    budget_split = args.budget_split

    print("Starting now...")

    #get current timestamp in date and time and create a string as 04.09.2024_12_34
    now = datetime.datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M")
    print("timestamp =", timestamp)

    # define CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    print("Using Device: ", os.environ["CUDA_VISIBLE_DEVICES"])

    proc = Popen(
        args=[
            'python', "program_refactoring/agent/agent_main.py",
            '--train_json_path',  'python_data/date/gpt-3.5-turbo_NL+SL/train_combined.jsonl',
            '--train_log_path', '/ceph/tsesterh/abstraction/regal_program_learning/data/test_runs_release/date/date_gpt35_main_agent_round1_new_12_seed',
            #'--train_log_path', '/ceph/tsesterh/abstraction/regal_program_learning',
            '--test_path', 'python_data/date/gpt-3.5-turbo_NL+SL/test.jsonl',
            #'--model_name', 'codellama/CodeLlama-7b-Instruct-hf' ,
            '--model_name', 'codellama/CodeLlama-13b-Instruct-hf',
            '--out_dir',  f'test_out/exp_{timestamp}_{exp_type}_{seed}_seed',
            #'--logdir',
            '--seed', f'{seed}',
            '--task', 'python', 
            '--dataset',  'date', 
            #'--use_success', 
            '--use_thought', 
            '--max_budget', f'{max_budget}',
            '--budget_split', f'{budget_split}',
            #'--batch_size', '1',
            #'--filter',  #TODO Tobi I made it false, otherwise would have been an issue with codebank
        ],
    )
    proc.wait()