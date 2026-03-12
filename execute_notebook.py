import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

def run_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    globals_dict = {}
    
    # Pre-populate globals if needed
    globals_dict['__name__'] = '__main__'
    
    cell_idx = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell_idx += 1
            print(f"\n--- RUNNING CELL {cell_idx} ---")
            
            # Combine lines of code
            code = "".join(cell['source'])
            
            # Replace paths for saving images
            code = code.replace('/mnt/user-data/outputs/', 'outputs/')
            
            try:
                # Execute the code in the globals_dict
                exec(code, globals_dict)
                print(f"Cell {cell_idx} completed successfully.")
            except Exception as e:
                print(f"ERROR IN CELL {cell_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Optionally continue or stop? Usually stop if a cell fails.
                # In this case, we'll stop to let the user know.
                # sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_notebook(sys.argv[1])
    else:
        run_notebook('logifly_case_study.ipynb')
