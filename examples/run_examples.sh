#!/bin/bash

# Run all examples (each example is a folder in the current directory to cd into), stop on error and print when it errors
for example in $(ls -d */); do
    echo "Running example: $example"
    cd $example
    if ! cpet.py -o options/options.json > cpet.out; then
        echo "Error running example: $example"
        exit 1
    else
        echo "Successfully ran example: $example"
        #Remove any output files or log files
        rm -f outdir/*.out
        rm -f cpet.out
    fi
    cd ..
done