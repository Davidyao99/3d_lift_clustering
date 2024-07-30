#!/bin/bash

# Path to your Python script
python_script="download-scannet.py"

# Input text file with line-separated items
input_file="scannetv2_val.txt"
output_folder="data_val"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file '$input_file' not found."
    exit 1
fi

# Read the input file line by line and process each item
while IFS= read -r item; do
    echo "Processing item: $item"
    yes "" | python "$python_script" -o $output_folder --id "$item"
done < "$input_file"