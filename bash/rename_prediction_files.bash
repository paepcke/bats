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

# Loop through all files with .log extension
for file in *.log; do
    # Check if file exists (this prevents errors if no .log files are found)
    if [ -f "$file" ]; then
        # Get the base name of the file (without the extension)
        base_name=$(basename "$file" .log)
        
        # Rename the file, replacing .log with .csv
        mv "$file" "${base_name}.csv"
        
        echo "Renamed $file to ${base_name}.csv"
    fi
done

echo "File renaming in $dir complete."
