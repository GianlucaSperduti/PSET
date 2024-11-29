from phon_utility.save_and_load import PickleLoader
import argparse

def from_dict_to_sentences(dictionary: dict, final_path:str, mode:str = 'a'):
    s = []
    for _, item in dictionary.items():
        for i in item:
            s.append(i)
    
    with open(final_path,  mode=mode) as f:
        for i in s:
            f.write(i)

def write_list_to_file(lst, final_path):
        print(f"List length: {len(lst)}")
        with open(final_path, mode='w') as f:
            for idx, item in enumerate(lst):
                if '\n' not in item:
                    item = item + '\n'
                else:
                    item = item
                f.write(str(item))
                if (idx + 1) % 1000 == 0:  # Log progress for every 1000 items
                    print(f"{idx + 1} items written to file.")
        with open(final_path, 'r') as f:
            lines = f.readlines()
            print(f"Number of lines in the file: {len(lines)}")
            # Print the first and last 10 lines to check content
            print("First 10 lines:")
            print("".join(lines[:10]))
            print("Last 10 lines:")
            print("".join(lines[-10:]))

def main():
    parser = argparse.ArgumentParser(description='From dictionary to sentences')
    parser.add_argument('--dictionary_path', type=str, help='Path to the dictionary file.')
    parser.add_argument('--plain_txt_dict_path', type=str, help='Path to the final file.')
    args = parser.parse_args()

    loader = PickleLoader()
    dictionary = loader.load(args.dictionary_path)

    new_list = []
    
    for key, item in dictionary.items():
        for i in item:
            new_list.append(i)

    print(len(new_list))
    print(f"List length: {len(new_list)}")

    write_list_to_file(new_list, args.plain_txt_dict_path)

if __name__ == '__main__':
    main()