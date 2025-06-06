import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial

# Define paths
data_dir = "/data/shipengcheng/Dataset/CVPR-BiomedSegFM/3D_train_npz_all"
challenge_data_dir = "/data/shipengcheng/code/CVPR2025_Text_guided_seg/SAT_cvpr2025challenge_0501/data/challenge_data"
output_file = challenge_data_dir + "/train_all.jsonl"
text_prompts_json = challenge_data_dir + "/CVPR25_TextSegFMData_with_class.json"

# Load text prompts
with open(text_prompts_json, 'r') as f:
    text_prompts = json.load(f)
    text_prompts_dataset_names = list(text_prompts.keys())

# Function to read existing entries and maintain sorted order
def load_existing_entries(filename):
    entries = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    # Sort by 'data' field
    return sorted(entries, key=lambda x: x['data'])

def process_file(file_path, existing_paths, text_prompts):
    if not file_path.endswith('.npz'):
        return None
        
    # Extract metadata
    modality = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    dataset = os.path.basename(os.path.dirname(file_path))

    dataset_list = list(text_prompts.keys())
    
    if dataset not in dataset_list:
        # print("dataset: ", dataset)
        return None
        
    # Skip if already exists
    if file_path in existing_paths:
        return None
    
    # Load NPZ file
    try:
        npz_data = np.load(file_path, allow_pickle=True)
        gts = npz_data['gts']           
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Process labels
    classes_list = list(text_prompts[dataset].keys())
    instance_label = text_prompts[dataset]["instance_label"]
    
    if "instance_label" in classes_list:
        classes_list.remove("instance_label")
    
    classes_list = [int(i) for i in classes_list]

    # num_classes = len(classes_list)

    classes_list_with_bg = [0] + classes_list
    if not np.isin(gts, classes_list_with_bg).all() and classes_list == [1]:
        print("dataset:", dataset)
        print("np.unique(gts):", np.unique(gts))
        print("classes_list_with_bg:", classes_list_with_bg)

        # Extract all arrays (assuming 'gts' is one of them)
        data_dict = {key: npz_data[key] for key in npz_data.files}
        
        # Modify gts: value>0 --> 1
        data_dict['gts'][data_dict['gts'] > 0] = 1

        gts = data_dict['gts']
        
        # Save back to .npz (overwrite)
        np.savez(file_path, **data_dict)
        
        if not np.isin(gts, classes_list_with_bg).all():
            print(f"No valid labels found in {file_path} after modification.")
            exit()
    else:
        pass

    # Create entry
    return {
        "data": file_path,
        "dataset": dataset,
        "modality": modality,
        "instance_label": instance_label
    }

def main():
    # Load existing entries
    existing_entries = load_existing_entries(output_file)
    existing_paths = {entry['data'] for entry in existing_entries}

    # Collect all files to process
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        print("Processing directory:", root)
        for file in files:
            if file.endswith('.npz'):
                all_files.append(os.path.join(root, file))

    # Process files in parallel
    num_processes = 12
    print(f"Processing {len(all_files)} files using {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_file, existing_paths=existing_paths, text_prompts=text_prompts)
        results = list(tqdm(pool.imap(process_func, all_files), total=len(all_files)))
    
    # Filter out None results
    new_entries = [entry for entry in results if entry is not None]

    # Merge and sort all entries
    all_entries = existing_entries + new_entries
    all_entries_sorted = sorted(all_entries, key=lambda x: x['data'])

    # Write back to file (this overwrites the existing file with sorted entries)
    with open(output_file, 'w') as out_f:
        for entry in all_entries_sorted:
            out_f.write(json.dumps(entry) + '\n')

    print(f"JSONL file updated with {len(new_entries)} new entries. Total entries: {len(all_entries_sorted)}")

if __name__ == '__main__':
    main()