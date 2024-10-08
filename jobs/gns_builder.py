import sys
import argparse

# =====================================================
# =*= CODE TO GENERATE JOBS FOR THE GNS SOLVER =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

START_IDX = 151
END_IDX = 200
SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']

'''
    TEST WITH
    python gns_builder.py --account=x --parent=y --mail=x@mail.com --time=20 --memory=187 --cpu=16
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII job builder")
    parser.add_argument("--account", help="Compute Canada Account", required=True)
    parser.add_argument("--parent", help="Compute Canada Parent Account", required=True)
    parser.add_argument("--mail", help="Compute Canada Email Adress", required=True)
    parser.add_argument("--time", help="Computing time", required=True)
    parser.add_argument("--memory", help="Computing RAM", required=True)
    parser.add_argument("--cpu", help="Computing CPUs", required=True)
    args = parser.parse_args()
    BASIC_PATH = "/home/"+args.account+"/projects/def-"+args.parent+"/"+args.account+"/gns/"

    f = open("./scripts/train_gns.sh", "w+")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --nodes 1\n")
    f.write(f"#SBATCH --time={args.time}:00:00\n")
    f.write(f"#SBATCH --mem={args.memory}G\n")
    f.write(f"#SBATCH --cpus-per-task={args.cpu}\n")
    f.write(f"#SBATCH --account=def-{args.parent}\n")
    f.write(f"#SBATCH --mail-user={args.mail}\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --mail-type=FAIL\n")
    f.write(f"#SBATCH --output={BASIC_PATH}data/out/train_gns.out\n")  
    f.write("module load python/3.12\n")
    f.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
    f.write("source $SLURM_TMPDIR/env/bin/activate\n")
    f.write("pip install --upgrade pip --no-index\n")
    f.write("pip install --no-index -r "+BASIC_PATH+"requirements.txt\n")
    f.write(f"python {BASIC_PATH}gns_solver.py --train=true --mode=prod --path="+BASIC_PATH+" \n")
    f.write("deactivate\n")
    f.close()
