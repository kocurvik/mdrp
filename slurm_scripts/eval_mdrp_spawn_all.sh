#!/bin/bash

# Check if directory path is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset_directory_path> <job_suffix>"
    echo "Example: $0 /home/kocurvik/data/mdrp/phototourism 001"
    exit 1
fi

DATASET_DIR="$1"
# Remove trailing slash if present
DATASET_DIR="${DATASET_DIR%/}"
# Get the base name of the dataset directory for naming
DATASET_TYPE=$(basename "$DATASET_DIR")
SUFFIX="$2"

# Create output directories if they don't exist
mkdir -p /home/kocurvik/logs/mdrp/${DATASET_TYPE}
mkdir -p /home/kocurvik/jobs/mdrp/${DATASET_TYPE}

# Template for individual job scripts
cat > job_template.sh << 'EOL'
#!/bin/bash
#SBATCH --account YOUR_ACCOUNT_NAME_HERE
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -G 0
#SBATCH -o /home/kocurvik/logs/mdrp/_DATASET_TYPE_/_JOB_NAME_.std.out
#SBATCH -e /home/kocurvik/logs/mdrp/_DATASET_TYPE_/_JOB_NAME_.err.out
#SBATCH --time=12:00:00

source /home/kocurvik/.bashrc
mamba init
mamba activate mdrp
cd /home/kocurvik/code/mdrp
export PYTHONPATH=/home/kocurvik/code/mdrp

# These will be replaced for each job
_DATASET_
_DATASET_PATH_

python eval.py -t 2.0 -r 16.0 -a -o -nw 64 $dataset_path/$dataset
python eval_shared_f.py -t 2.0 -r 16.0 -a -o -nw 64 $dataset_path/$dataset
python eval_varying_f.py -t 2.0 -r 16.0 -a -o -nw 64 $dataset_path/$dataset

python eval.py -g -t 2.0 -r 16.0 -a -o -nw 64 $dataset_path/$dataset
python eval_shared_f.py -g -t 2.0 -r 16.0 -a -o -nw 64 $dataset_path/$dataset
python eval_varying_f.py -g -t 2.0 -r 16.0 -a -o -nw 64 $dataset_path/$dataset

EOL

# Function to create and submit a job
create_and_submit_job() {
    local dataset=$1
    local suffix=$2
    local job_name="eval_${dataset}.${suffix}"
    
    # Create specific job script
    local job_script="/home/kocurvik/jobs/mdrp/${DATASET_TYPE}/${job_name}.sh"
    cp job_template.sh "$job_script"
    
    # Replace placeholders
    sed -i "s|_DATASET_PATH_|dataset_path=\"$DATASET_DIR\"|" "$job_script"
    sed -i "s|_DATASET_TYPE_|${DATASET_TYPE}|" "$job_script"
    sed -i "s|_JOB_NAME_|${job_name}|" "$job_script"
    sed -i "s|_DATASET_|dataset=\"$dataset\"|" "$job_script"
    
    # Make script executable
    chmod +x "$job_script"
    
    # Submit job
    sbatch --job-name="$job_name" "$job_script"
}

# Check if directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Directory $DATASET_DIR does not exist"
    exit 1
fi

# Submit one job per dataset (each job will run both versions)
for dataset in $(ls "$DATASET_DIR"); do
    create_and_submit_job "$dataset" "$SUFFIX"
done

# Clean up template
rm job_template.sh
