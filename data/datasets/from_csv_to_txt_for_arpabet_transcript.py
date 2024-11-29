import csv
import argparse

def convert_csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        words = [word for row in reader for word in row]

    with open(txt_file, 'w') as file:
        for word in words:
            file.write(word + '\n')

def from_csv_to_txt_for_arpabet_transcript(csv_file_path, txt_file_path):
    convert_csv_to_txt(csv_file_path, txt_file_path)

def main():
    parser = argparse.ArgumentParser(description='Convert csv file to txt file for arpabet transcript')
    parser.add_argument('--csv_path', type=str, help='Path to csv file')
    parser.add_argument('--txt_path', type=str, help='Path to txt file')
    args = parser.parse_args()
    from_csv_to_txt_for_arpabet_transcript(args.csv_path, args.txt_path)

if __name__ == '__main__':
    main()