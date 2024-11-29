import re 
import warnings
from phon_utility.save_and_load import PickleLoader, PickleSaver

def clean_file(path):
    # Read the entire file's content
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Clean each line
    cleaned_lines = [line.replace('{', '').replace('}', '') for line in lines]
    
    # Write the cleaned content back to the file, overwriting it
    with open(path, 'w') as f:
        f.writelines(cleaned_lines)

def read_and_strip_dataset(dataset):
    with open(dataset, 'r') as f:
        dataset = f.readlines()
        dataset = [line.strip() for line in dataset]
        dataset = [line.split(',') for line in dataset]
        dataset_final = []
        for lines in dataset:
            for line in lines:
                dataset_final.append(line)
    return dataset_final

def check_if_header(listed_dataset):
    to_remove = ['a', 'b', 'c', 'd']
    first_three_items = listed_dataset[:3]
    # Count how many items from 'to_remove' are not in the first 3 items
    missing_count = sum(letter not in first_three_items for letter in to_remove)
    # If at least 3 letters from 'to_remove' are not in the first 3 positions
    if missing_count >= 3:
        warnings.warn(
            'Are you sure you set the header correctly? Remember to set the header as "a", "b", "c", ... and so on.'
        )


def remove_header(listed_dataset):
    check_if_header(listed_dataset)
    to_remove = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    for header in to_remove:
        if header in listed_dataset:
            listed_dataset.remove(header)
    return listed_dataset

def mapping_ipa_dict_to_grapheme_dict(ipa_dataset, grapheme_dataset):
    grapheme_name, ipa_name = grapheme_dataset.split('/')[-1].split('.csv')[0], ipa_dataset.split('/')[-1].split('.csv')[0]
    if contains_sentence(grapheme_name, ipa_name):
        ipa_dataset = remove_header(read_and_strip_dataset(ipa_dataset))
        grapheme_dataset = remove_header(read_and_strip_dataset(grapheme_dataset))
        if len(ipa_dataset) != len(grapheme_dataset):
            raise ValueError("IPA and grapheme datasets should have the same length.")
        ipa_to_grapheme_dict = {}
        for ipa, grapheme in zip(ipa_dataset, grapheme_dataset):
            if ipa in ipa_to_grapheme_dict.keys():
                ipa_to_grapheme_dict[ipa].append(grapheme)
            else: 
                grapheme_list = []
                grapheme_list.append(grapheme)
                ipa_to_grapheme_dict[ipa] = grapheme_list
    else:
        raise ValueError("IPA and grapheme datasets should have the same name.")    
    return ipa_to_grapheme_dict

def check_errors_in_args(args):
    if hasattr(args, 'load_ipa_paths') and hasattr(args, 'IPA_path'):
        if args.load_ipa_paths and args.IPA_path:
            raise ValueError("You cannot provide both IPA loading paths and an IPA output path, such as you need to extract them.")
    if hasattr(args, 'load_arpa_paths') and hasattr(args, 'ARPA_path'):
        if args.load_arpa_paths and args.ARPA_path:
            raise ValueError("You cannot provide both ARPABET loading paths and a txt output path, such as you need to extract them.")
    if "Phoneme2Vec" in args.selected_models and not args.p2v_model:
        raise ValueError("You need to provide a path to the Phoneme2Vec model.")
    if args.load_ipa_paths or args.load_arpa_paths:
        if len(args.clean_dataset_paths) != len(args.load_ipa_paths) or len(args.clean_dataset_paths) != len(args.load_arpa_paths):
            raise ValueError("You need to provide the same number of paths for the clean datasets and their phoentized versions.")
    if args.articulatory_embs_do_not_need_format_correction and not "ArticulatoryPhonemes" in args.selected_models:
        raise ValueError("You did not select Articulatory Embeddings but you want to skip the format correction for Art. Embs. Reformulate your request.")
    if args.skip_edit_distance_test:
        warnings.warn("Skipping edit distance test.")
    if args.do_not_extract_embeddings:
        warnings.warn("Skipping embeddings extraction.")
    if args.articulatory_embs_do_not_need_format_correction:
        warnings.warn("Skipping articulatory embeddings format correction.")

def creating_unique_formatting_keys(ipa_to_grapheme_dict):
    # Sometimes it may happen that the same IPA pronunciation has more than one graphic correspondence.
    # This function handles this eventuality.
    final_set = set()
    for key, item in ipa_to_grapheme_dict.items():
        if len(item) > 1:
            counter = 0
            for _ in item:
                counter += 1
                final_set.add(key + '_' + str(counter))
        else:
            final_set.add(key)
    return final_set
    
def handling_mismatch(embs_dict, ipa_to_grapheme_dict):
    unique_formatting_keys = creating_unique_formatting_keys(ipa_to_grapheme_dict)
    unique_embeddings_keys = set(embs_dict.keys())
    missing_keys = unique_formatting_keys - unique_embeddings_keys
    extra_keys = unique_embeddings_keys - unique_formatting_keys

    if missing_keys or extra_keys:
        if missing_keys:
            print(f"Warning: Missing embeddings for these IPA keys: {missing_keys}")
        if extra_keys:
            print(f"Warning: Extra embeddings found for these keys: {extra_keys}")
        raise ValueError(
            f"Discrepancy in dictionary lengths: The formatted embeddings dictionary has {len(unique_formatting_keys)} entries "
            f"while the original embeddings dictionary has {len(unique_embeddings_keys)} entries. "
            f"Possible issues: Missing IPA keys ({missing_keys}) or extra embeddings ({extra_keys})."
        )

def embs_format_correction(embs_dict_path, ipa_to_grapheme_dict):
    formatted_embs_dict = {}
    loader = PickleLoader()
    embs_dict = loader.load(embs_dict_path)
    #handling_mismatch(embs_dict, ipa_to_grapheme_dict)
    for key, item in embs_dict.items():
        if len(ipa_to_grapheme_dict[key]) > 1:
            for grapheme in ipa_to_grapheme_dict[key]:
                formatted_embs_dict[grapheme] = item
        else:
            formatted_embs_dict[ipa_to_grapheme_dict[key][0]] = item
    return formatted_embs_dict

def articulatory_embs_format_correction(embs_dict_path, ipa_dataset, grapheme_dataset, output_path):
    ps = PickleSaver()
    ipa_to_grapheme_dict = mapping_ipa_dict_to_grapheme_dict(ipa_dataset, grapheme_dataset)
    formatted_embs_dict = embs_format_correction(embs_dict_path, ipa_to_grapheme_dict)
    ps.save(formatted_embs_dict, output_path)
    return formatted_embs_dict

def check_if_no_problems_in_dataset_names(clean_paths):
    if '/' in clean_paths or '.' in clean_paths:
        raise ValueError("Dataset names should not contain '/' or '.' characters. Are you sure the pipeline is correctly set?")

def contains_sentence(x, y):
    # Escape any special characters in x to ensure they are treated as literals
    escaped_x = re.escape(x)
    
    # Search for the escaped_x pattern in y
    # re.IGNORECASE makes the search case-insensitive if needed
    match = re.search(escaped_x, y, re.IGNORECASE)
    
    # If there's a match, return True; otherwise, return False
    return match is not None

def check_if_clean_paths_in_any_embs(clean_paths, full_embeddings):
    does_clean_path_match = []
    for clean_path in clean_paths:
        for full_embedding in full_embeddings:
            if contains_sentence(clean_path, full_embedding):
                print(f"Found a match for {clean_path} in {full_embedding}.")
                does_clean_path_match.append(True)
    if len(does_clean_path_match) != len(full_embeddings):
        raise ValueError("No matches found between embedding and dataset names. Are you sure the pipeline is correctly set?")
            
def assign_correct_dataset_to_correct_embs(clean_dataset_paths, full_embeddings):
    clean_paths_ordered = []
    full_embeddings_ordered = []
    clean_paths_names = [path.split('/')[-1].split('.csv')[0] for path in clean_dataset_paths]
    clean_path_equivalences = {}
    for clean_path_name, clean_path in zip (clean_paths_names, clean_dataset_paths):
        clean_path_equivalences[clean_path_name] = clean_path
    check_if_no_problems_in_dataset_names(clean_paths_names)
    check_if_clean_paths_in_any_embs(clean_paths_names, full_embeddings)
    for clean_path_name in clean_paths_names:
        for full_embedding in full_embeddings:
            if contains_sentence(clean_path_name, full_embedding):
                print(f"Assigning {clean_path_name} to {full_embedding}")
                clean_paths_ordered.append(clean_path_equivalences[clean_path_name])
                full_embeddings_ordered.append(full_embedding)
    
    return clean_paths_ordered, full_embeddings_ordered


