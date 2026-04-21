import pickle
import os

def get_structure(obj, indent=0):
    """Recursively crawls the object to report types and lengths."""
    spacing = "  " * indent
    obj_type = type(obj).__name__
    
    # Handle List or Tuple
    if isinstance(obj, (list, tuple)):
        size = len(obj)
        print(f"{spacing}- {obj_type} (length: {size})")
        if size > 0:
            # Check the first element as a sample of the nested structure
            print(f"{spacing}  [Sample of first element]:")
            get_structure(obj[0], indent + 2)
            
    # Handle Dictionary
    elif isinstance(obj, dict):
        print(f"{spacing}- {obj_type} (keys: {len(obj)})")
        for key in obj.keys():
            val = obj[key]
            val_type = type(val).__name__
            # If the value is a container, dive deeper
            if isinstance(val, (dict, list, tuple)):
                print(f"{spacing}  Key '{key}': {val_type} (length: {len(val)})")
                # Dive one level deeper for nested containers
                if len(val) > 0:
                    get_structure(val, indent + 3)
            else:
                print(f"{spacing}  Key '{key}': {val_type}")
    
    # Handle Basic Types
    else:
        print(f"{spacing}- {obj_type}")

def check_pickle_schema(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"--- Pickle Schema Report: {file_path} ---")
        get_structure(data)

    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
import argparse
parser = argparse.ArgumentParser(description='Check Pickle File Schema')
parser.add_argument('pkl_file', type=str, help='Path to the pickle file to check')
args = parser.parse_args()
file_to_open = args.pkl_file
check_pickle_schema(file_to_open)