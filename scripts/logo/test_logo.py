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
    #add argument for gpus
    parser.add_argument("--gpus", type=str, default="5")
    #batch size
    parser.add_argument("--batch_size", type=int, default=5)

    #add argument for number of samples
    parser.add_argument("--n_samples", type=int, default=2)

    args = parser.parse_args()

    # make n_samples an environment variable
    os.environ["N_SAMPLES"] = str(args.n_samples)

    #print all arguments in a nice way
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    exp_type = args.exp_type
    seed = args.seed
    max_budget = args.max_budget
    budget_split = args.budget_split
    gpus = args.gpus
    batch_size = args.batch_size

    if exp_type == "use_abstraction":
        train_log = "/ceph/tsesterh/abstraction/regal_program_learning/data/test_runs_release/logo/logo_gpt35_main_agent_yes_refactor_5_yes_filter_5_yes_retry_yes_comment_helpers_second_gpt35_round_filter_12_seed"
    elif exp_type == "no_abstraction":
        train_log = "/ceph/tsesterh/abstraction/regal_program_learning"
    else:
        raise ValueError("exp_type must be 'use_abstraction' or 'no_abstraction'")

    print("Starting now...")

    #get current timestamp in date and time and create a string as 04.09.2024_12_34
    now = datetime.datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M")
    print("timestamp =", timestamp)

    # define CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus if gpus else "0"
    print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])

    print("Using Device: ", os.environ["CUDA_VISIBLE_DEVICES"])

    proc = Popen(
        args=[
            'python', "program_refactoring/agent/agent_main.py",
            '--train_json_path',  'logo_data/python/train_200.jsonl',
            '--train_log_path', f'{train_log}',
            #'--train_log_path', '/ceph/tsesterh/abstraction/regal_program_learning',
            '--test_path', 'logo_data/python/test.jsonl',
            '--model_name', 'codellama/CodeLlama-7b-Instruct-hf', #'tsesterh/codellama_7b_instruct_logo', #'codellama/CodeLlama-7b-Instruct-hf', #'codellama/CodeLlama-70b-Instruct-hf', #'deepseek-ai/DeepSeek-Coder-V2-Instruct-0724', #'deepseek-ai/deepseek-coder-7b-instruct-v1.5', ', # 'xu3kev/deepseekcoder-7b-logo-pbe',
            '--out_dir',  f'test_out/logo/exp_{timestamp}_{exp_type}_{seed}_seed',
            #'--logdir',
            '--seed', f'{seed}',
            '--task', 'logos', 
            '--dataset',  'logos', 
            #'--use_success', 
            '--use_thought', 
            '--max_budget', f'{max_budget}',
            '--budget_split', f'{budget_split}',
            '--batch_size', f'{batch_size}',
            #'--filter',  #TODO Tobi I made it false, otherwise would have been an issue with codebank
        ],
    )
    proc.wait()