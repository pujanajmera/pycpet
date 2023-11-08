import os

# Specify the directory containing the files
directory = './'

# List all files in the directory
files = os.listdir(directory)
print("num files: ", len(files))
# Iterate through the files and rename them
for filename in files:
    # Check if the file matches the desired format
    if filename.startswith('out.pdb.'):
        # Extract the integer from the filename
        file_number = filename.split('.')[2]
        
        # Create the new file name using the extracted integer
        new_filename = 'frame_{}.pdb'.format(file_number)
        
        # Get the full file paths for the old and new names
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)
        
        print(f'Renamed {filename} to {new_filename}')
