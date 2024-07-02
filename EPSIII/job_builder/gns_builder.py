import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII job builder")
    parser.add_argument("--account", help="Compute Canada Account", required=True)
    parser.add_argument("--parent", help="Compute Canada Parent Account", required=True)
    parser.add_argument("--mail", help="Compute Canada Email Adress", required=True)
    args = parser.parse_args()
    BASIC_PATH = "/home/"+args.account+"/projects/def-"+args.parent+"/"+args.account+"/GNS/"
    f = open("../jobs/gns.sh", "w+")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --nodes 1\n")
    f.write("#SBATCH --time=10:00:00\n")
    f.write("#SBATCH --cpus-per-task=64\n")
    f.write("#SBATCH --mem=16G\n")
    f.write("#SBATCH --account=def-"+args.parent+"\n")
    f.write("#SBATCH --mail-user="+args.mail+"\n")
    f.write("#SBATCH --mail-type=FAIL\n")
    f.write("#SBATCH --output="+BASIC_PATH+"EPSIII/out/gns.out\n")  
    f.write("module load python/3.12\n")
    f.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
    f.write("source $SLURM_TMPDIR/env/bin/activate\n")
    f.write("pip install --upgrade pip --no-index\n")
    f.write("pip install --no-index -r "+BASIC_PATH+"requirements.txt\n")
    f.write("python "+BASIC_PATH+"EPSIII/gns_solver.py\n")
    f.write("deactivate\n")
    f.close()