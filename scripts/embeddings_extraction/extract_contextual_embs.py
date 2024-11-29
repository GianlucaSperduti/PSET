from embeddings.embeddings_models import XPhoneBERT, ClassicBERT, KeyContextExtractor
from ipa_extraction.IpaExtractor import IpaTranscriptionSentence
from phon_utility.save_and_load import PickleLoader, PickleSaver
from tqdm import tqdm
import torch
import os
import glob
import argparse


def get_embeddings(model, w_to_s: dict, embs_path: str, batch_size: int=3, transcriptor = False, window_size:int=20, windows_reduction_anyway=False):
    # If sentence is longer than the model max length, reduced to n tokens (n is window size)
    key_context_extractor = KeyContextExtractor(model_max_length=512, tokenizer = model.tokenizer, window=window_size, windows_reduction_anyway=windows_reduction_anyway)
    batch_count = 0
    embs_dict = {}
    for key, item in tqdm(w_to_s.items()):
        key_for_context = key
        batch_count += 1
        embs_list = []
        for s in item:
            # If transcriptor is not False, it should be a function that takes a list of strings and return it transcribed in IPA alphabet
            if transcriptor:
                # key is the central word that we want to extract the context from
                key_for_context = transcriptor([key])
                key_for_context = key_for_context[0]
            # s is the sentence in which the key is present. If sentence is longer than the model max length, reduced to n tokens (n is window size)
            s = key_context_extractor.check_sentence(s, key_for_context)
            s = s.replace('▁','')
            embeddings = model.embed_from_sentence(s, key_for_context)
            embs_list.append(embeddings)
        embs_dict[key] = embs_list
        if batch_count % batch_size == 0:
            path = os.path.join(embs_path, 'batch_' + str(batch_count) + '.pkl')
            PickleSaver.save(embs_dict, path)
            embs_dict = {}

def delete_batch_files(embs_path:str):
     #delete the other files in the folder
    for file in glob.glob(os.path.join(embs_path, '*.pkl')):
        filename = os.path.basename(file)
        if 'batch' in filename:
            os.remove(file)

def torch_simple_list_mean(single_emb):
    single_emb = torch.mean(single_emb, dim=0)
    return single_emb
        

def classic_bert_average(single_emb):
    new_emb = []
    if len(single_emb.shape) == 3:
        single_emb =  torch_simple_list_mean(single_emb[0])
        new_emb.append(single_emb)
    elif len(single_emb.shape) == 2:
        single_emb = torch_simple_list_mean(single_emb)
        new_emb.append(single_emb)
    else:
        new_emb.append(single_emb)
    return new_emb[0]

def xphone_bert_average(single_emb):
    new_emb = []
    if len(single_emb.shape) == 3:
        single_emb =  torch_simple_list_mean(single_emb[0])
        new_emb.append(single_emb)
    elif len(single_emb.shape) == 2:
        single_emb = torch_simple_list_mean(single_emb)
        new_emb.append(single_emb)
    else:
        new_emb.append(single_emb)
    return new_emb[0]

def final_tensor_average(embs_dict:dict, model:str):
    final_embs = {}
    for key, item in embs_dict.items():
        new_item = []
        for single_emb in item:
            if model == 'ClassicBERT':
                single_emb = classic_bert_average(single_emb)
                new_item.append(single_emb)
            elif model == 'XPhoneBERT':
                single_emb = xphone_bert_average(single_emb)
                new_item.append(single_emb)
            else:
                raise ValueError('Model not supported')
                    
        final_embs[key] = torch.mean(torch.stack(new_item), dim=0)
    return final_embs
        
def save_final_embeddings(embs_path:str, model:str, file_name:str='final'):
    # It should load all the pkl in the folder and merge them in a single file
    embeddings = {}
    for file in glob.glob(os.path.join(embs_path, '*.pkl')):
        embs_dict = PickleLoader.load(file)
        embeddings.update(embs_dict)
        
    embeddings = final_tensor_average(embeddings, model)
    merged_path = os.path.join(embs_path, f'{file_name}.pkl')
    PickleSaver.save(embeddings, merged_path)
    delete_batch_files(embs_path)



def main(args):
        dataset_name = args.dataset_name
        w_to_s = PickleLoader.load(args.abc_path)
        embs_path = args.output_folder

        if args.transcriptor:
            transcriptor = IpaTranscriptionSentence().generate_phonetic_transcriptions
        else: 
            transcriptor = False
        
        models = {'ClassicBERT': ClassicBERT(), 
        'XPhoneBERT': XPhoneBERT()}
        model = models[args.model]

        if args.only_save:
            save_final_embeddings(embs_path, args.model, file_name=f'{dataset_name}_{model.__class__.__name__}')
        else:
            get_embeddings(model, w_to_s, embs_path, transcriptor=transcriptor, window_size=args.window_size, windows_reduction_anyway=args.windows_reduction_anyway)
            save_final_embeddings(embs_path, args.model, file_name=f'{dataset_name}_{model.__class__.__name__}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='abc file name (e.g., anchors, homophones or synonyms or others).')
    parser.add_argument('--abc_path', type=str, help='abc path')
    parser.add_argument('--transcriptor', type=bool, default=False, help='If True, the words are transcribed in IPA alphabet')
    parser.add_argument('--model', type=str, help='Model to use for extracting embeddings')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--window_size', type=int, default=20, help='Window size')
    parser.add_argument('--only_save', type=bool, default=False, help='If True, only save the final embeddings')
    parser.add_argument('--windows_reduction_anyway', type=bool, default=False, help='If True, the sentence is reduced to n tokens (n is window size) anyway')
    args = parser.parse_args()
    main(args)