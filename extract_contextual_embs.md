# Contextual Embeddings Extraction

## IMPORTANT INSTRUCTIONS

1. **ID**: Essential at every step. Do not change IDs or file names; it will cause errors.
2. **Folders**: Always use the same working folder specified in the second script (`--contextual_emb_folder`). Do not change hard-coded folder names.

## Extracting Contextual Embeddings

### 1. Extracting Sentences

- **Script**: `extract_sentences_with_specific_words.py`
- **Inputs**:
  - `id`: Identifying ID needed.
  - `dataset`: Dataset from Huggingface (e.g., Wikitext).
  - `split`: Split of the dataset (e.g., train).
  - `n`: Number of sentences per word.
  - `log_path`: Path to the log file (.txt preferred).
  - `save_path`: Output path (.pkl preferred).
- **Output**: Dictionary `{word: [list of sentences]}`.

### 2. Manual Completion

- **Script**: `manually_complete_missing_sentences.py`

- **Arguments**:
  - `--contextual_emb_folder`: Folder where the phonetic embeddings will be saved.
  - `--Id`: The ID extracted in phase 1. This ID is used to load the correct file for preparation and to export the final version.
  - `--prepare`: Prepares a `.txt` file that needs to be MANUALLY completed with new sentences to finalize the lists.
  - `--load_and_export`: Loads the manually completed file and exports the final `.pkl` dictionary containing the complete `{word: [list of sentences]}`.

- **Process**:
  1. **Prepare**: Run the script with the `--prepare` argument to generate a text file listing missing sentences. This file must be manually updated with additional sentences.
  2. **Load and Export**: After manually completing the text file with new sentences, run the script again with the `--load_and_export` argument. This will load the updated file and produce the final dictionary with all sentences for each word.

- **Note**: This step usually requires manual intervention to ensure that every word in the dictionary has a sufficient number of sentences. If the dictionary is not completed properly, the entire process will be flawed, resulting in inconsistent contextual information for some words. This is the format before the manual processing: "contextual_embs/id_number.txt". You will find entries like this ('lint', 6). 
You have to completely modify the txt. Suppose you decided for n = 10. 6 means that we have already 6 sentences in the 
original dataset. You should add 4 sentences that include 'lint':
so you should delete this: 
('lint', 6)
and go on with this:
lint, (word who's missing separated by comma)
linting the lint, (all the sentences needed)
lint lint lint,
lint,
linting the lintable lint

- 

### 3. Phonetic Transcription

- **Script**: `from_grapheme_dicts_to_phoneme_dicts.py`
- **Arguments**:
  - `--dictionary_path`: Output from the `load_and_export` phase.
  - `--plain_txt_dict_path`: File with plain text of words and sentences.
  - `--transcriptions_path`: File with sentences translated into IPA.
  - `--path_final_transcribed_dictionary`: Path to the final dictionary with IPA.

### 4. Extracting Contextual Embeddings

- **Script**: `extract_contextual_embs.py`
- Uses the `.pkl` files (both IPA and non-IPA) generated in previous phases.

## Notes

Make sure to follow these steps carefully and check each phase to avoid errors.
