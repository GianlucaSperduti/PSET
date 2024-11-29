from phon_utility.save_and_load import PickleLoader,PickleSaver
import argparse
import torch

def stock_and_average_embs(embs):
    stacked_tensors = torch.stack(embs)
    average_tensor = stacked_tensors.mean(dim=0)
    return average_tensor

def main():
    parser = argparse.ArgumentParser(description='Averaging of contextual embeddings')
    parser.add_argument('--contextual_embs_dictionary', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    args = parser.parse_args()

    contextual_embs_dictionary = PickleLoader.load(args.contextual_embs_dictionary)
    averaged_embs = {}
    for key, value in contextual_embs_dictionary.items():
        pre_averaged_embs = []
        for m_value in value:

            # This is very important: sometimes our reference token is divided in two or more tokens, so for this reason
            # We need to average two times
            if len(m_value) > 1:
                m_value = m_value.mean(dim=0).reshape(1, -1)
            if len(m_value.shape) > 2:
                m_value = m_value.mean(dim=1)
            if m_value.shape[0] == 0:
                continue


            pre_averaged_embs.append(m_value)

        average_tensor = stock_and_average_embs(pre_averaged_embs)
        averaged_embs[key] = average_tensor
    
    PickleSaver.save(averaged_embs, args.output_file)

if __name__ == '__main__':
    main()