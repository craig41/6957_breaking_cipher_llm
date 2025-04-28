#!/bin/bash
# Script to calculate average XCOMET score from an output file
# Usage: ./calculate_xcomet_average.sh <output_file>

WORKDIR=/scratch/general/vast/$USER/6957_breaking_cipher_llm
source $WORKDIR/.venv/bin/activate

# Check if argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_file>"
    echo "Example: $0 outputs-4124354"
    exit 1
fi

OUTPUT_FILE=$1

# If OUTPUT_FILE is just a filename (not a path), prepend the eval directory
if [[ ! -f "$OUTPUT_FILE" && -f "$WORKDIR/eval/$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="$WORKDIR/eval/$OUTPUT_FILE"
fi

# Check if OUTPUT_FILE exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Output file doesn't exist: $OUTPUT_FILE"
    exit 1
fi


# Run the Python script
echo "Calculating average XCOMET score from $OUTPUT_FILE"
python $WORKDIR/eval/calculate_average_xcomet.py "$OUTPUT_FILE"

echo "Done!"