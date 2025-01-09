#!/usr/bin/env python3

import os
import sys
import json
import yaml
import argparse
import datetime
import subprocess
import cv2
import numpy as np
import torch
import faiss

from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from typing import List
from tqdm import tqdm

from prompts import llama_prompt_fork, llama_prompt_intro_fork


#set TOKENIZERS_PARALLELISM=False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


###############################################################################
# TRAIN FUNCTION
###############################################################################
def train_lora_adapter(
    base_model_name: str,
    support_examples: List[dict],
    lora_output_dir: str,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    max_seq_length: int
):
    """
    Trains a LoRA adapter on top of a base model + local support data.
    Returns the trained model in memory (not merged). 
    """

    # Build dataset from support examples
    train_dataset = Dataset.from_list(support_examples)

    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=lora_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        max_steps=steps,
        optim="paged_adamw_32bit",
        fp16=True,
        report_to="none"
    )

    # SFTTrainer from the trl library
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Actual training
    trainer.train()

    # Save the LoRA adapter 
    trainer.model.save_pretrained(os.path.join(lora_output_dir, "final_checkpoint"))

    # Return the trained model so we can do immediate inference in-memory
    return trainer.model, trainer


###############################################################################
# IN-MEMORY INFERENCE (AFTER TRAINING) HELPER
###############################################################################
def run_inference_on_single_example_in_memory(
    model,
    tokenizer,
    device,
    query_text: str,
    retrieval_context: str = "",
    max_new_tokens: int = 200
):
    """
    Runs inference using an *already-loaded* model in memory (model with LoRA).
    This is used immediately after training a LoRA on a single test sample.
    """
    if retrieval_context:
        final_prompt = f"{llama_prompt_intro_fork}{retrieval_context}# Query:\n{query_text}\n"
    else:
        final_prompt = llama_prompt_fork.format(query=query_text)

    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_prompt, generated_text


###############################################################################
# DISK-BASED INFERENCE FUNCTION (Bulk)
###############################################################################
def run_inference(
    base_model_name: str,
    inference_output_dir: str,
    retrieval_data: List[dict],
    index,
    embedding_model,
    tokenizer,
    device,
    test_data,
    top_k_prompt: int,
    use_base_model: bool,
    retrieval_in_prompt: bool,
    lora_dir: str = None
):
    """
    Performs inference for each test example by either:
      - Using the base model only, or
      - Loading the base model + LoRA adapter from disk

    This is the "old" version that loads from disk each time.
    """
    results = []

    # If requested, load the base model once (no LoRA).
    if use_base_model:
        print("Using Base Model (NO LoRA) for Inference.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model_base.config.use_cache = False
        model_base = prepare_model_for_kbit_training(model_base)
    else:
        model_base = None

    for i, test_ex in tqdm(enumerate(test_data)):
        input_text = test_ex["prompt"]

        # Optionally prepend nearest-neighbor examples to the prompt
        if retrieval_in_prompt:
            test_emb = embedding_model.encode([input_text], convert_to_numpy=True)
            faiss.normalize_L2(test_emb)
            if top_k_prompt > 0:
                distances, indices = index.search(test_emb, top_k_prompt)
                support_examples = [retrieval_data[idx] for idx in indices[0]]
            else:
                support_examples = []
            context_str = ""
            for ex in support_examples:
                context_str += f"# Example Prompt:\n{ex['prompt']}\n# Example Completion:\n{ex['completion']}\n\n"

            final_prompt = f"{llama_prompt_intro_fork}{context_str}# Query:\n{input_text}\n"
            prompt = final_prompt
        else:
            prompt = llama_prompt_fork.format(query=input_text)

        if use_base_model:
            # Inference with pure base model
            model = model_base
        else:
            # Load base model + LoRA from disk
            final_checkpoint = os.path.join(lora_dir, f"lora_adapter_test_{i}", "final_checkpoint")
            if not os.path.exists(final_checkpoint):
                print(f"LoRA adapter path does not exist: {final_checkpoint}")
                continue

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(model)

            # Attach LoRA
            model = get_peft_model(model, LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=8,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"]
            ))
            model.load_adapter(final_checkpoint, adapter_name="default_lora_adapter")

        model.to(device)

        # Generate output
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            temperature=0.7
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "index": i,
            "prompt": prompt,
            "generated_text": generated_text
        })

        # If we are not using the base model globally, free memory after each iteration
        if not use_base_model:
            del model
            torch.cuda.empty_cache()

    # Save inference results
    os.makedirs(inference_output_dir, exist_ok=True)
    results_file = os.path.join(inference_output_dir, "inference_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Inference finished and saved to {results_file}.")
    return results_file


###############################################################################
# EVALUATION FUNCTION
###############################################################################
def evaluate_generated_programs(inference_results_path: str, evaluation_output_dir: str):
    """
    Evaluates generated Python programs by executing them, rendering the resulting 
    image, and then comparing to a gold image from the test set.
    """
    os.makedirs(evaluation_output_dir, exist_ok=True)

    start_prog = """from program_refactoring.domains.logos.pyturtle import PyTurtle
from program_refactoring.domains.logos.pyturtle import HALF_INF, INF, EPS_DIST, EPS_ANGLE

turtle = PyTurtle()
def forward(dist):
    turtle.forward(dist)    
def left(angle):
    turtle.left(angle)
def right(angle):   
    turtle.right(angle)
def teleport(x, y, theta):
    turtle.teleport(x, y, theta)
def penup():
    turtle.penup()
def pendown():
    turtle.pendown()
def position():
    return turtle.x, turtle.y
def heading():
    return turtle.heading
def isdown():
    return turtle.is_down
def embed(program, local_vars):
    return turtle.embed(program, local_vars)
def fork_state():
    return turtle.fork_state()
"""

    end_prog = """turtle.save('{hf_path}/{infer}_{i}.jpg')"""

    # Load inference results
    try:
        json_data = json.load(open(inference_results_path, 'r'))
    except FileNotFoundError:
        print(f"[WARN] Could not open inference file {inference_results_path}.")
        return

    # Load test dataset (path can be adjusted if needed)
    data = []
    test_dataset_path = "/ceph/tsesterh/abstraction/regal_program_learning/logo_data/python/test_dataset_instruct.jsonl"
    with open(test_dataset_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    def vis_compare(img1, img2):
        """
        Pixel-based comparison:
         - All fully white pixels are ignored.
        """
        if img1 is None or img2 is None:
            return 0.0

        ones = np.ones(img1.shape)
        ones1 = ones.copy()
        ones1[img1 == 255] = 0
        ones2 = ones.copy()
        ones2[img2 == 255] = 0

        idxs1 = set(np.flatnonzero(ones1).tolist())
        idxs2 = set(np.flatnonzero(ones2).tolist())

        pixels_nonzero_both = idxs1 | idxs2

        match_pixels = np.equal(img1, img2)
        match_pixels = match_pixels.reshape(-1)[np.array(list(pixels_nonzero_both), dtype=np.int64)]
        num_matches = np.sum(match_pixels)

        return num_matches / len(pixels_nonzero_both) if len(pixels_nonzero_both) > 0 else 0.0

    correctnesses = []
    accuracies = []
    results_summary = []

    head_file_part = os.path.dirname(inference_results_path)

    for i, d in enumerate(json_data):
        task = d['prompt']
        solution = d['generated_text']
        index = d['index']

        # Remove the prompt text from the solution
        solution = solution.replace(task, "").strip()

        # If "save" is found in the last lines => remove them
        lines = solution.split("\n")
        while lines and "save" in lines[-1]:
            lines.pop()
        solution = "\n".join(lines)

        # Paths for code and images
        pred_prog_path = os.path.join(head_file_part, f'logos_{index}_pred.py')
        pred_img_path = os.path.join(head_file_part, f'result_{index}.jpg')
        gold_prog_path = os.path.join(head_file_part, f'logos_{index}_gold.py')
        gold_img_path = os.path.join(head_file_part, f'gold_{index}.jpg')

        # Build the prediction program
        prog_pred = (
            start_prog
            + "\n"
            + solution
            + "\n"
            + end_prog.format(hf_path=head_file_part, infer="result", i=index)
        )

        # Remove special tokens
        for token in ["[PYTHON]", "[/PYTHON]", "[INST]", "[/INST]"]:
            prog_pred = prog_pred.replace(token, "")

        with open(pred_prog_path, 'w') as f:
            f.write(prog_pred)

        # Execute the prediction program
        p = subprocess.Popen(["python", pred_prog_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, errs = p.communicate()
        out, errs = out.decode(), errs.decode()
        print(out)
        if errs:
            print(errs)
            results_summary.append({
                "index": index,
                "task": task,
                "correctness": None,
                "accuracy": None,
                "status": "no_solution"
            })
            continue

        if not os.path.exists(pred_img_path):
            results_summary.append({
                "index": index,
                "task": task,
                "correctness": None,
                "accuracy": None,
                "status": "no_solution"
            })
            continue

        pred_img = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)

        # Retrieve gold program from test dataset
        target = data[i]
        target_prog = target['completion'].strip()

        prog_gold = (
            start_prog
            + "\n"
            + target_prog
            + "\n"
            + end_prog.format(hf_path=head_file_part, infer="gold", i=index)
        )

        with open(gold_prog_path, 'w') as f:
            f.write(prog_gold)

        # Execute gold program
        p = subprocess.Popen(["python", gold_prog_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, errs = p.communicate()
        out, errs = out.decode(), errs.decode()
        print(out)
        if errs:
            print(errs)
            results_summary.append({
                "index": index,
                "task": task,
                "correctness": None,
                "accuracy": None,
                "status": "no_solution"
            })
            continue

        if not os.path.exists(gold_img_path):
            results_summary.append({
                "index": index,
                "task": task,
                "correctness": None,
                "accuracy": None,
                "status": "no_solution"
            })
            continue

        target_img = cv2.imread(gold_img_path, cv2.IMREAD_GRAYSCALE)
        print("[INFO] Gold image loaded.")

        # Compare images
        accuracy = vis_compare(target_img, pred_img)
        correctness = bool(accuracy > 0.95)

        correctnesses.append(correctness)
        accuracies.append(accuracy)
        status = "success" if correctness else "failure"

        results_summary.append({
            "index": index,
            "task": task,
            "correctness": correctness,
            "accuracy": accuracy,
            "status": status
        })
        print("[INFO] Checked a solution.")

    # Summarize
    eval_file_path = os.path.join(evaluation_output_dir, "results_summary.json")
    with open(eval_file_path, "w") as outfile:
        json.dump(results_summary, outfile, indent=4)

    # Quick stats
    if correctnesses:
        avg_correctness = sum(c for c in correctnesses if c is not None) / len(correctnesses)
    else:
        avg_correctness = 0

    if accuracies:
        avg_accuracy = sum(a for a in accuracies if a is not None) / len(accuracies)
    else:
        avg_accuracy = 0

    print(f"[INFO] Correctness Rate: {avg_correctness:.3f}")
    print(f"[INFO] Average Accuracy: {avg_accuracy:.3f}")
    print(f"[INFO] Evaluation file written to: {eval_file_path}")


###############################################################################
# MAIN
###############################################################################
def main():
    """
    Main function that reads a config.yaml to decide whether to run train/infer/eval
    and writes all relevant results to the appropriate folders.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Extract important fields from config.yaml
    do_train = config.get("train", False)
    do_infer = config.get("infer", False)
    do_eval = config.get("eval", False)
    base_model_name = config.get("model_name", "codellama/CodeLlama-7b-Instruct-hf")
    dataset_name = config.get("dataset_name", "tsesterh/logo_data_instruct_fork")
    top_k_train = config.get("top_k_train", 20)
    top_k_prompt = config.get("top_k_prompt", 20)
    lora_rank = config.get("lora_rank", 8)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.05)
    max_seq_length = config.get("max_seq_length", 1024)
    finetune_steps = config.get("finetune_steps", 100)
    use_non_finetuned = config.get("use_base_model", False)
    retrieval_in_prompt = config.get("retrieval_in_prompt", False)

    # Timestamp
    date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Model storage directory
    model_folder_name = f"{base_model_name.replace('/', '_')}_{top_k_train}"
    save_model_dir = os.path.join("models", model_folder_name)
    os.makedirs(save_model_dir, exist_ok=True)

    # Write a copy of the config.yaml for reference
    config_copy_path = os.path.join(save_model_dir, f"config_{date_str}.yaml")
    with open(config_copy_path, 'w') as cf:
        yaml.dump(config, cf)

    # Load dataset
    dataset = load_dataset(dataset_name)
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    # Combine train+val for retrieval
    retrieval_data = [ex for ex in train_data] + [ex for ex in val_data]
    retrieval_prompts = [ex["prompt"] for ex in retrieval_data]

    # Build FAISS index
    print("[INFO] Loading SentenceTransformer for retrieval embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    retrieval_embeddings = embedding_model.encode(retrieval_prompts, convert_to_numpy=True, show_progress_bar=True)
    d = retrieval_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(retrieval_embeddings)
    index.add(retrieval_embeddings)

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # TRAIN
    if do_train:
        print("[INFO] Starting training per test example with Top-K Nearest Neighbors...")
        # We only do immediate inference on a single test item if do_infer is also True.
        for i, test_ex in enumerate(test_data):
            test_prompt = test_ex["prompt"]

            # Retrieve top-K neighbors
            test_emb = embedding_model.encode([test_prompt], convert_to_numpy=True)
            faiss.normalize_L2(test_emb)
            distances, indices = index.search(test_emb, top_k_train)
            support_examples = [retrieval_data[idx] for idx in indices[0]]

            # Format support examples
            support_examples = [
                {"prompt": llama_prompt_fork.format(query=ex["prompt"]), "completion": ex["completion"]}
                for ex in support_examples
            ]

            # LoRA output directory
            lora_output_dir = os.path.join(save_model_dir, f"lora_adapter_test_{i}")
            os.makedirs(lora_output_dir, exist_ok=True)

            # Train => returns the model in memory
            trained_model, trainer = train_lora_adapter(
                base_model_name=base_model_name,
                support_examples=support_examples,
                lora_output_dir=lora_output_dir,
                steps=finetune_steps,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_seq_length=max_seq_length
            )

            print(f"[INFO] Training complete for test example {i}, saved to {lora_output_dir}.")

            # If we also want inference right after training
            if do_infer:
                # Build the retrieval context for the single test sample, if needed
                context_str = ""
                if retrieval_in_prompt:
                    for ex in support_examples:
                        context_str += f"# Example Prompt:\n{ex['prompt']}\n# Example Completion:\n{ex['completion']}\n\n"

                # Move model to device
                trained_model.to(device)

                # Inference on the single test example
                final_prompt, generated_text = run_inference_on_single_example_in_memory(
                    model=trained_model,
                    tokenizer=tokenizer,
                    device=device,
                    query_text=test_prompt,
                    retrieval_context=context_str,
                    max_new_tokens=200
                )

                # Save partial results
                inference_dir = os.path.join("inference", f"{base_model_name.replace('/', '_')}_topktrain{top_k_train}_topkprompt{top_k_prompt}_sampled1")
                os.makedirs(inference_dir, exist_ok=True)
                results_file = os.path.join(inference_dir, "inference_results.json")

                # If the file already exists, load it to append results
                if os.path.exists(results_file):
                    with open(results_file, "r", encoding="utf-8") as f:
                        old_results = json.load(f)
                else:
                    old_results = []

                # Append new result
                old_results.append({
                    "index": i,
                    "prompt": final_prompt,
                    "generated_text": generated_text
                })

                # Write back
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(old_results, f, ensure_ascii=False, indent=2)

                print(f"[INFO] Immediate inference for test example {i} completed. Appended to {results_file}")

                # Evaluate if requested
                if do_eval:
                    evaluate_generated_programs(
                        inference_results_path=results_file,
                        evaluation_output_dir=inference_dir
                    )
                    print(f"[INFO] Evaluation for test example {i} completed.")

            # Release resources for this iteration
            del trained_model
            del trainer
            torch.cuda.empty_cache()

        print("[INFO] All training runs completed.")

    # If not training but user only wants inference (bulk inference from disk):
    elif do_infer:
        infer_folder_name = f"{base_model_name.replace('/', '_')}_topktrain{top_k_train}_topkprompt{top_k_prompt}_sampled1"
        inference_dir = os.path.join("inference", infer_folder_name)
        os.makedirs(inference_dir, exist_ok=True)

        inference_results_path = run_inference(
            base_model_name=base_model_name,
            inference_output_dir=inference_dir,
            retrieval_data=retrieval_data,
            index=index,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            device=device,
            test_data=test_data,
            top_k_prompt=top_k_prompt,
            use_base_model=use_non_finetuned,
            retrieval_in_prompt=retrieval_in_prompt,
            lora_dir=save_model_dir
        )

        # Evaluate if requested
        if do_eval:
            evaluate_generated_programs(
                inference_results_path=inference_results_path,
                evaluation_output_dir=inference_dir
            )
    else:
        print("[INFO] Neither train nor infer requested. Nothing to do.")


if __name__ == "__main__":
    main()