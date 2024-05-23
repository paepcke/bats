import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i")
parser.add_argument("--output", "-o")

args = parser.parse_args()

source_directory = args.input
destination_directory = args.output

# Function to group directories by year and month and create symlinks
def copy_grouped_directories(source, destination):
    # List of directories in the source directory (excluding files)
    dir_names = [name for name in os.listdir(source) if os.path.isdir(os.path.join(source, name))]

    # Sort directory names
    sorted_dir_names = sorted(dir_names)

    # Group directories by every 4 months
    grouped_dirs = [sorted_dir_names[i:i+4] for i in range(0, len(sorted_dir_names), 4)]

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination):
        os.makedirs(destination)

    for group in grouped_dirs:
        # Determine the group name by the first and last element in each group
        group_name = f"{group[0]}_to_{group[-1]}"
        group_path = os.path.join(destination, group_name)
        
        # Create a directory for the group
        if not os.path.exists(group_path):
            os.makedirs(group_path)
        
        # Copy each directory in the group to the new location
        for dir_name in group:
            source_path = os.path.join(source, dir_name)
            destination_path = os.path.join(group_path, dir_name)
            
            # Copy directory to the new location
            shutil.copytree(source_path, destination_path)
# Ensure the destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Run the function to create grouped directories with symlinks
copy_grouped_directories(source_directory, destination_directory)
