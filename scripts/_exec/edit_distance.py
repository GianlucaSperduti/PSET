import argparse
import subprocess

def edit_distance(csv_paths):
    for csv_path in csv_paths:
        process = subprocess.run(['python', '/home/sperduti/phonetic_embeddings/scripts/edit_distance_calc.py',
                                  '--csv_path', csv_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            print(f"Error processing {csv_path}: {process.stderr.decode('utf-8')}")
        else:
            print(f"Processed {csv_path} successfully: {process.stdout.decode('utf-8')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run edit distance calculation on multiple CSV files.")
    parser.add_argument('csv_paths', metavar='CSV', type=str, nargs='+', 
                        help='paths to CSV files to process')
    
    args = parser.parse_args()
    edit_distance(args.csv_paths)
