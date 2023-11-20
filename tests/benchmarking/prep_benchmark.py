#!/usr/bin/env python3

import os
import shutil

def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)

    # Parse the folder name
    parts = folder_name.split('_')
    charges, samples, method, cores_requested = parts[:4]
    print(charges)
    concurrent_streamlines = parts[4] if len(parts) > 4 else None

    # Determine the correct files to copy based on the method
    if method == 'mcpet':
        shutil.copy('options_mcpet_generic.txt', folder_path)
        shutil.copy('submit_mCPET.slurm', folder_path)
        options_file = os.path.join(folder_path, 'options_mcpet_generic.txt')
    elif method == 'pycpet':
        shutil.copy('options_pycpet_generic.json', folder_path)
        shutil.copy('submit_pyCPET_CPU.slurm', folder_path)
        options_file = os.path.join(folder_path, 'options_pycpet_generic.json')

    # Copy the charges file
    shutil.copy(f'{charges}_charges.pqr', folder_path)

    # Edit the options file
    with open(options_file, 'r') as file:
        content = file.read()

    # Replace placeholders based on 'charges'
    if charges == '4k':
        print("yes")
        # Add substitutions for 4k here
        content=content.replace('center1','34.806')
        content=content.replace('center2','37.859')
        content=content.replace('center3','18.465')
        content=content.replace('x1','35.292')
        content=content.replace('x2','39.838')
        content=content.replace('x3','18.873')
        content=content.replace('y1','36.456')
        content=content.replace('y2','38.407')
        content=content.replace('y3','17.249')
    elif charges == '20k':
        # Add substitutions for 20k here
        content=content.replace('center1','54.283')
        content=content.replace('center2','10.211')
        content=content.replace('center3','48.321')
        content=content.replace('x1','53.743')
        content=content.replace('x2','10.093')
        content=content.replace('x3','49.155')
        content=content.replace('y1','53.593')
        content=content.replace('y2','9.707')
        content=content.replace('y3','47.803')
    elif charges == '31k':
        # Add substitutions for 31k here
        content=content.replace('center1','104.785')
        content=content.replace('center2','113.388')
        content=content.replace('center3','117.966')
        content=content.replace('x1','105.785')
        content=content.replace('x2','113.388')
        content=content.replace('x3','117.966')
        content=content.replace('y1','104.785')
        content=content.replace('y2','114.388')
        content=content.replace('y3','117.966')
    content=content.replace("{charges}",f"{charges}")
    # Convert samples to actual numbers
    samples_number = {'1k': 1000, '10k': 10000, '100k': 100000}[samples]
    if method == 'mcpet':
        content = content.replace('samples N', f'samples {samples_number}')
    elif method == 'pycpet':
        content = content.replace('"n_samples": N,', f'"n_samples": {samples_number},')

    # Replace concurrent_streamlines if applicable
    if concurrent_streamlines and method == 'pycpet':
        content = content.replace('"concur_slip": M,', f'"concur_slip": {concurrent_streamlines},')

    # Write the updated content back to the options file
    with open(options_file, 'w') as file:
        file.write(content)

    # Edit the .slurm file
    slurm_file = os.path.join(folder_path, 'submit_mCPET.slurm' if method == 'mcpet' else 'submit_pyCPET_CPU.slurm')
    with open(slurm_file, 'r') as file:
        slurm_content = file.read()

    slurm_content = slurm_content.replace('N_CORES_REQUESTED', cores_requested)
    slurm_content = slurm_content.replace('PQR_FILE', f'{charges}_charges.pqr')

    with open(slurm_file, 'w') as file:
        file.write(slurm_content)

# Process each subfolder in the CPU directory
cpu_path = './CPU'
for folder in os.listdir(cpu_path):
    full_folder_path = os.path.join(cpu_path, folder)
    if os.path.isdir(full_folder_path):
        process_folder(full_folder_path)

