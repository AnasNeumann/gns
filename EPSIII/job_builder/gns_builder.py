import sys
import argparse

START_IDX = 151
END_IDX = 200
SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII job builder")
    parser.add_argument("--account", help="Compute Canada Account", required=True)
    parser.add_argument("--parent", help="Compute Canada Parent Account", required=True)
    parser.add_argument("--mail", help="Compute Canada Email Adress", required=True)
    args = parser.parse_args()
    BASIC_PATH = "/home/"+args.account+"/projects/def-"+args.parent+"/"+args.account+"/GNS/"

    # Training job
    f = open("../jobs/train_gns.sh", "w+")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --nodes 1\n")
    f.write("#SBATCH --cpus-per-task=16\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --time=10:00:00\n")
    f.write("#SBATCH --mem=64G\n")
    f.write("#SBATCH --account=def-"+args.parent+"\n")
    f.write("#SBATCH --mail-user="+args.mail+"\n")
    f.write("#SBATCH --mail-type=FAIL\n")
    f.write("#SBATCH --output="+BASIC_PATH+"EPSIII/out/train_gns.out\n")  
    f.write("module load python/3.12\n")
    f.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
    f.write("source $SLURM_TMPDIR/env/bin/activate\n")
    f.write("pip install --upgrade pip --no-index\n")
    f.write("pip install --no-index -r "+BASIC_PATH+"requirements.txt\n")
    f.write("python "+BASIC_PATH+"EPSIII/gns_solver.py --train=1 --mode=prod\n")
    f.write("deactivate\n")
    f.close()

    # Testing jobs
    for size in SIZES:
        for instance in range(START_IDX, END_IDX+1):
            f = open("../jobs/gns/"+str(size)+"/"+str(instance)+".sh", "w+")
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --nodes 1\n")
            f.write("#SBATCH --time=1:00:00\n")
            f.write("#SBATCH --mem=64G\n")
            f.write("#SBATCH --cpus-per-task=16\n")
            f.write("#SBATCH --gres=gpu:1\n")
            f.write("#SBATCH --account=def-"+args.parent+"\n")
            f.write("#SBATCH --mail-user="+args.mail+"\n")
            f.write("#SBATCH --mail-type=FAIL\n")
            f.write("#SBATCH --output="+BASIC_PATH+"EPSIII/out/gns_"+str(size)+"_"+str(instance)+".out\n")  
            f.write("module load python/3.12\n")
            f.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
            f.write("source $SLURM_TMPDIR/env/bin/activate\n")
            f.write("pip install --upgrade pip --no-index\n")
            f.write("pip install --no-index -r "+BASIC_PATH+"requirements.txt\n")
            f.write("python "+BASIC_PATH+"EPSIII/gns_solver.py --train=0 --mode=prod --size="+size+" --id="+str(instance)+"\n")
            f.write("deactivate\n")
            f.close()
