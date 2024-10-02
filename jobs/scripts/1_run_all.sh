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
        "exact_s")
            echo "Running small instances with exact solver..."
            slurm "./exact/s"
            ;;
        "exact_m")
            echo "Running medium instances with exact solver..."
            slurm "./exact/m"
            ;;
        "exact_l")
            echo "Running large instances with exact solver..."
            slurm "./exact/l"
            ;;
        "exact_xl")
            echo "Running extra-large instances with exact solver..."
            slurm "./exact/xl"
            ;;
        "exact_xxl")
            echo "Running XXL instances with exact solver..."
            slurm "./exact/xxl"
            ;;
        "exact_xxxl")
            echo "Running 3XL instances with exact solver..."
            slurm "./exact/xxxl/"
            ;;
        "gns_s")
            echo "Running small instances with GNS solver..."
            slurm "./gns/s"
            ;;
        "gns_m")
            echo "Running medium instances with GNS solver..."
            slurm "./gns/m"
            ;;
        "gns_l")
            echo "Running large instances with GNS solver..."
            slurm "./gns/l"
            ;;
        "gns_xl")
            echo "Running extra-large instances with GNS solver..."
            slurm "./gns/xl"
            ;;
        "gns_xxl")
            echo "Running XXL instances with GNS solver..."
            slurm "./gns/xxl"
            ;;
        "gns_xxxl")
            echo "Running 3XL instances with GNS solver..."
            slurm "./gns/xxxl/"
            ;;
        "train")
            echo "Running the unique training GNS job..."
            slurm "./train_gns.sh"
            ;;
    esac
done
