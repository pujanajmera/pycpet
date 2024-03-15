#!/usr/bin/env python3

import os
import shutil

def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)

    # Parse the folder name
    parts = folder_name.split('_')
    charges, samples, step_size, index = parts[:4]

    # Determine the correct files to copy based on the method
    shutil.copy('options_mcpet_generic.txt', folder_path)
    shutil.copy('submit_mCPET.slurm', folder_path)
    options_file = os.path.join(folder_path, 'options_mcpet_generic.txt')

    # Copy the charges file
    shutil.copy(f'4k_charges.pqr', folder_path)

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
    # Convert samples to actual numbers
    content = content.replace('samples N', f'samples {samples}')
    content = content.replace('stepSize N', f'stepSize {step_size}')
    step_size_string = step_size[2:]
    content = content.replace('sampleOutput NAME', f'sampleOutput {charges}_{samples}_{step_size_string}_{index}') 
    # Write the updated content back to the options file
    with open(options_file, 'w') as file:
        file.write(content)

    # Edit the .slurm file
    slurm_file = os.path.join(folder_path, 'submit_mCPET.slurm')
    with open(slurm_file, 'r') as file:
        slurm_content = file.read()

    slurm_content = slurm_content.replace('N_CORES_REQUESTED', '32')
    slurm_content = slurm_content.replace('PQR_FILE', f'{charges}_charges.pqr')

    with open(slurm_file, 'w') as file:
        file.write(slurm_content)

# Process each subfolder in the CPU directory
cpu_path = './mcpet_convergence'
for folder in os.listdir(cpu_path):
    full_folder_path = os.path.join(cpu_path, folder)
    if os.path.isdir(full_folder_path):
        process_folder(full_folder_path)

