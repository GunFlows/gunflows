

# open an npz file, inspect its contents, and print the keys and shapes of the arrays inside
import numpy as np
import sys  
from pathlib import Path

def inspect_npz(file_path: str):
    # Load the .npz file
    data = np.load(file_path)

    print(f"Inspecting file: {file_path}")
    print("Contents:")
    
    # Iterate through the arrays in the .npz file
    for key in data.files:
        array = data[key]
        print(f"Key: {key}, Shape: {array.shape}, Dtype: {array.dtype}")

    # print list of key "par_names"
    if "par_names" in data.files:
        par_names = data["par_names"]
        print("\nParameter Names:")
        for name in par_names:
            print(name.decode('utf-8') if isinstance(name, bytes) else name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_npz.py <path_to_npz_file>")
        sys.exit(1)

    npz_file_path = sys.argv[1]
    if not Path(npz_file_path).is_file():
        print(f"Error: File '{npz_file_path}' does not exist.")
        sys.exit(1)

    inspect_npz(npz_file_path)