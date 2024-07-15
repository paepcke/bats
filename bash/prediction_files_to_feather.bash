#!/usr/bin/env bash

# Check if a directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get the directory from the command line argument
dir="$1"

# Check if the provided directory exists
if [ ! -d "$dir" ]; then
    echo "Error: Directory '$dir' does not exist."
    exit 1
fi

# Change to the specified directory
cd "$dir" || exit 1

export PYTHONPATH=/home/paepcke/EclipseWorkspaces/feather-tools/src:$PYTHONPATH

# Loop through all files with .log extension
for file in *.csv; do
    # Check if file exists (this prevents errors if no .log files are found)
    if [ -f "$file" ]; then
        # Get the base name of the file (without the extension)
        base_name=$(basename "$file" .csv)

    	/home/paepcke/EclipseWorkspaces/feather-tools/src/feather_tools/csv2f $file
        echo "Created ${base_name}.feather"
    fi
done

echo "File convertion to .feather in $dir complete."
