# ===================================================================
# =*= CODE TO GENERATE JOBS TO CONSTRUCT THE PRE-TRAINING DATASET =*=
# ===================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

START_IDX = 1
END_IDX = 100
SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']

if __name__ == '__main__':
    f = open("../dataset_builder.sh", "w+")
    f.write("#!/bin/bash\n")
    f.write("python3.12 -m venv gns_env\n")
    f.write("source gns_env/bin/activate\n")
    f.write("pip3.12 install --upgrade pip\n")
    f.write("pip3.12 install -r requirements.txt\n")
    for size in SIZES:
        for i in range(START_IDX, END_IDX):
            f.write(f"python3.12 gns_solver.py --size={size} --id={i} --train=false --mode=prod --number=0 --savestates=true --path=./ \n")
    f.write("deactivate\n")
    f.close()
