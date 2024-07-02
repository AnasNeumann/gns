#!/bin/bash
args=("$@")

slurm() {
    folder=$1
    for f in "$folder"/*.sh; do
        sbatch "$f"
    done
}

for arg in "${args[@]}"; do
    case $arg in
        "s")
            echo "Running small instances..."
            slurm "./exact/s"
            ;;
        "m")
            echo "Running medium instances..."
            slurm "./exact/m"
            ;;
        "l")
            echo "Running large instances..."
            slurm "./exact/l"
            ;;
        "xl")
            echo "Running extra-large instances..."
            slurm "./exact/xl"
            ;;
        "xxl")
            echo "Running XXL instances..."
            slurm "./exact/xxl"
            ;;
        "xxxl")
            echo "Running 3XL instances..."
            slurm "./exact/xxxl/"
            ;;
        "gns")
            echo "Running the unique GNS job..."
            slurm "./gns.sh"
            ;;
    esac
done
