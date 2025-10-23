#!/bin/bash

# Run all examples (each example is a folder in the current directory to cd into), stop on error and print when it errors. Only run examples with options file in the directory, otherwise skip
for example in $(ls -d */); do
    echo "Running example in debug mode: $example"
    cd $example
    if [ ! -f options/options.json ]; then
        echo "Skipping example (no options file found): $example"
        cd ..
        continue
    fi
    if ! cpet.py -d -o options/options.json >> ../run_examples.out 2>&1; then
        echo "Error running example: $example"
        exit 1
    else
        echo "Successfully ran example: $example"
        #Remove any output files or log files
        rm -f outdir/*.out
        # rm -f cpet.out
    fi
    cd ..
done
