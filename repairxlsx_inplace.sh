#!/bin/bash

# Define the directory where the xlsx files are located
dir="raw_data"

# Define the directory where the repaired files will be saved
repaired_dir="repaired_data"

# Define the number of parallel processes
processes=6

# Use find to search for all xlsx files recursively
find "$dir" -type f -name "*.xlsx" | while read file; do
  # Create the directory structure in the repaired directory
  relative_path="$(echo "$file" | sed "s|$dir||")"
  repaired_file_dir="$repaired_dir$(dirname "$relative_path")"
  mkdir -p "$repaired_file_dir"
  repaired_file="$repaired_file_dir/$(basename "$file")"
  # Skip the file if it has already been repaired
  if [ -f "$repaired_file" ]; then
    continue
  fi
  # If the number of parallel processes is less than the maximum, start a new process
  if [ "$(jobs | wc -l)" -lt "$processes" ]; then
    /Applications/LibreOffice.app/Contents/MacOS/soffice --headless --invisible --convert-to xlsx "$file" --outdir "$repaired_file_dir" &
  else
    # Wait for a process to finish before starting a new one
    wait
  fi
done

# Wait for all processes to finish
wait

# Copy all .txt files to the corresponding location in the repaired directory
find "$dir" -type f -name "*.txt" | while read file; do
  relative_path="$(echo "$file" | sed "s|$dir||")"
  repaired_file_dir="$repaired_dir$(dirname "$relative_path")"
  mkdir -p "$repaired_file_dir"
  repaired_file="$repaired_file_dir/$(basename "$file")"
  cp "$file" "$repaired_file"
done