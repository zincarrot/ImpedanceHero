#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: repair_xlsx.sh [input_folder] [output_folder]"
  exit 1
fi

input_folder=$1
output_folder=$2

if [ ! -d "$input_folder" ]; then
  echo "Error: input folder does not exist"
  exit 1
fi

if [ ! -d "$output_folder" ]; then
  echo "Creating output folder: $output_folder"
  mkdir "$output_folder"
fi

for file in "$input_folder"/*.xlsx; do
  filename=$(basename "$file")
  /Applications/LibreOffice.app/Contents/MacOS/soffice --headless --convert-to xlsx "$file" --outdir "$output_folder"
  echo "Repaired $filename and saved to $output_folder"
done
