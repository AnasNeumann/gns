# Engineer-To-Order (ETO) Graph Neural Scheduling (GNS) Project
A small Graph Attention Network (GAT) to schedule jobs in a ETO manufacturing environement, trained with the Proxmimal Policy gradient Optimization (PPO) reinforcement learning algorithm.

## Locally try the project
1. `python -m venv ./gns_env`
2. `source ./gns_env/bin/activate`
3. `pip install --upgrade -r requirements.txt`
4. CHOOSE EITHER GNS_SOLVER, EXACT_SOLVER, or INSTANCE_GENERATOR (_see bellow for the rest_)
5. `desactivate`

# Test the instance generator
```python
bash _env.sh
python instance_generator.py --train=150 --test=50
```

# Test the exact (Google OR-Tool) solver
```python
bash _env.sh
python exact_solver.py --size=s --id=151 --path=./
```

# Test the GNS solver in solving mode
```python
bash _env.sh
python gns_solver.py --size=s --id=151 --train=false --mode=test --path=./
```

# Test the GNS solver in training mode
```python
bash _env.sh
python gns_solver.py --train=true --mode=test --path=./
```

# Generate exact jobs for DRAC production
* _DRAC = Digital Research Alliance of Canada_
```python
# job duration, number of CPUs, and memory used are dynamically related to the instance size (no GPU/TPU for exact jobs)
python ./jobs/exact_builder.py --account=x --parent=y --mail=x@mail.com
```

# Generate the GNS training job for DRAC production
```python
python ./jobs/gns_builder.py --account=x --parent=y --mail=x@mail.com --time=10 --memory=187 --cpu=16
```

# Execute all jobs in DRAC production (_train GNN and solve testing instances with exact solver_)
```python
cd jobs/scripts/
bash 0_run_purge.sh
bash 1_run_all.sh train exact_s exact_m exact_l exact_xl exact_xxl exact_xxxl
```