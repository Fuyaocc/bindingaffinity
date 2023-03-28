#!/bin/bash
shopt -s nullglob
for file in ../cleandata/*; do
    filename=$(basename "$file")
    echo "$filename"
    python utils/renumber_pdb.py -i "$file"  -r  -c > "../pdbs/$filename"
done