#!/bin/bash

# Define the directory where the xlsx files are located
dir="raw_data"

# Define the directory where the repaired files will be saved
repaired_dir="repaired_data"

# Use find to search for all xlsx and txt files recursively
find "$dir" -type f \( -name "*.xlsx" -o -name "*.txt" \) | while read file; do
  # Create the directory structure in the repaired directory
  relative_path="$(echo "$file" | sed "s|$dir||")"
  repaired_file_dir="$repaired_dir$(dirname "$relative_path")"
  mkdir -p "$repaired_file_dir"
  # If the file is a txt file, copy it to the repaired directory
  if [ "${file##*.}" == "txt" ]; then
    cp "$file" "$repaired_file_dir"
  fi
done