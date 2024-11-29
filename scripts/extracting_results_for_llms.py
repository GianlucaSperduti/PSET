import json
from phon_utility.save_and_load import load_and_quadruplicate_dataset
from llms_results_extraction.llms_results_extraction import LLMSResultsExtraction
import numpy as np
import argparse
import logging
import pandas as pd

# Function to configure loggers
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler for writing to a file
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Stream handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

# Function to map GPT answers to corresponding columns
def map_llms_to_column(row):
    if row['llms_answer'].lower() == row['b'].lower():
        return 'b'
    elif row['llms_answer'].lower() == row['c'].lower():
        return 'c'
    elif row['llms_answer'].lower() == row['d'].lower():
        return 'd'
    elif row['llms_answer'].lower() == 'no_answer':
        return 'no_answer'
    return 'no_match'

# Function to calculate percentages of each unique answer
def calculate_unique_answer_percentage(original_dataset):
    value_counts = original_dataset['mapped_answers'].value_counts()
    percentage_counts = (value_counts / len(original_dataset['mapped_answers'])) * 100
    for answer, percentage in percentage_counts.items():
        print(f'{answer}: {percentage:.2f}%')
    return percentage_counts

def create_prompt_based_answers(data):
    # Initialize empty lists for each position in the quartets
    first_prompt = []
    second_prompt = []
    third_prompt = []
    fourth_prompt = []

    # Iterate over the values in the dictionary
    for _, quartet in data.items():
        # Append the first, second, third, and fourth elements to their respective lists
        first_prompt.append(quartet[0])
        second_prompt.append(quartet[1])
        third_prompt.append(quartet[2])
        fourth_prompt.append(quartet[3])

    # Return a list of quartets created from the elements at each position
    return [first_prompt, second_prompt, third_prompt, fourth_prompt]

def assign_mapped_dataset_to_prompt(original_dataset, textual_answers_by_prompt):
    all_the_datasets = {}
    c = 0
    for answers in textual_answers_by_prompt:
        original_dataset = original_dataset.copy()
        # Create a new column with the prompt
        original_dataset['llms_answer'] = answers
        original_dataset['mapped_answers'] = original_dataset.apply(map_llms_to_column, axis=1)
        all_the_datasets[c] = original_dataset
        c += 1
    return all_the_datasets

def creating_ablation_dataset(original_dataset, textual_answers_by_id):
        mapped_datasets_results = {}
        all_the_datasets = assign_mapped_dataset_to_prompt(original_dataset, textual_answers_by_id)
        for prompt_id, dataset in all_the_datasets.items():
            percentage_count = calculate_unique_answer_percentage(dataset)
            mapped_datasets_results[prompt_id] = percentage_count
        return mapped_datasets_results

def all_answers_unique_answer_percentage(original_dataset, textual_answers_by_id):
    # Assuming assign_mapped_dataset_to_id returns a dictionary of datasets
    all_the_datasets = assign_mapped_dataset_to_prompt(original_dataset, textual_answers_by_id)
    
    # Collect all datasets into a list
    all_answers = [dataset for _, dataset in all_the_datasets.items()]
    
    # Concatenate all datasets into a single DataFrame
    combined_answers = pd.concat(all_answers, axis=0, ignore_index=True)
    percentage_count = calculate_unique_answer_percentage(combined_answers)

    return percentage_count


# Main function
def main():
    parser = argparse.ArgumentParser(description='Extracting results for GPT-4')
    parser.add_argument('--llms_answer', type=str, help='The llms_answer file')
    parser.add_argument('--model', type=str, help='The model used for the response')
    parser.add_argument('--dataset_path', type=str, help='The dataset file')
    parser.add_argument('--output', type=str, help='The output file')
    args = parser.parse_args()

    # Set up two loggers: one for wrong answers, one for general results
    results_logger = setup_logger('results_logger', 'results_logfile.txt')
    wrong_answers_logger = setup_logger('wrong_answers_logger', 'wrong_answers_logfile.txt')

    # Load and quadruplicate the dataset
    original_dataset = pd.read_csv(args.dataset_path)
    quadruplicated_dataset = load_and_quadruplicate_dataset(args.dataset_path)

    # Load the input JSON file
    with open(args.llms_answer) as f:
        data = json.load(f)

    # Process llms responses
    LLMS_answer_extractor = LLMSResultsExtraction(wrong_answers_logger)
    n_correct_answers, label_answers_for_id, all_answers, textual_answers_by_id = LLMS_answer_extractor.extract_results(data, args.model)

    # PROMPT BASED ANALYSIS

    # Create prompt-based answers: this row is to extract 0 - 1 labels for each prompt. This is useful to get general statistics
    # (mean std) about each prompt considering the task as binary classification.
    label_prompt_based_results = create_prompt_based_answers(label_answers_for_id)
    prompt_based_mean = np.mean(label_prompt_based_results)
    prompt_based_std = np.std(label_prompt_based_results)

    # Output the number of correct answers
    results_logger.info(f'Correct answers: {n_correct_answers}/{len(data)}')
    results_logger.info(f'Prompt-based mean: {prompt_based_mean}')
    results_logger.info(f'Prompt-based std: {prompt_based_std} \n')

    # ABALTION ANALYSIS
    textual_answers_by_prompt = create_prompt_based_answers(textual_answers_by_id)
    mapped_datasets_results = creating_ablation_dataset(original_dataset, textual_answers_by_prompt)
    for prompt_id, result in mapped_datasets_results.items():
        results_logger.info(f'Prompt {prompt_id}: {result} \n')

    # ALL ANSWERS ANALYSIS
    percentage_count = all_answers_unique_answer_percentage(original_dataset, textual_answers_by_prompt)
    results_logger.info(f'Percentage of unique answers: {percentage_count} \n')

    # CHECK (to be deleted)
    quadruplicated_dataset['llms_answer'] = all_answers
    # Map the GPT answers to the corresponding columns
    quadruplicated_dataset['mapped_answers'] = quadruplicated_dataset.apply(map_llms_to_column, axis=1)
    percentage_count_check = calculate_unique_answer_percentage(quadruplicated_dataset)
    results_logger.info(f'Percentage of unique answers FROM QUADRUPLICATED DATASET: {percentage_count_check}')

    # Save the updated dataset to a CSV file
    if args.output:
        # Add llms answers to the dataset
        quadruplicated_dataset['llms_answer'] = all_answers
        # Map the GPT answers to the corresponding columns
        quadruplicated_dataset['mapped_answers'] = quadruplicated_dataset.apply(map_llms_to_column, axis=1)
        quadruplicated_dataset.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
