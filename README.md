# Engineer-To-Order (ETO) Graph Neural Scheduling (GNS) Project
A Graph Attention Network (GAT) to schedule jobs in an ETO manufacturing environment, trained with a Multi-Agent version of the Proximal Policy Optimization (MAPPO) algorithm.

## Refer to this repository in scientific documents
`Neumann, Anas (2024). A hyper-graph neural network trained with multi-agent proximal policy optimization to schedule projects in engineer-to-order environments *GitHub repository: https://github.com/AnasNeumann/gns*.`

```bibtex
    @misc{HGNS,
      author = {Anas Neumann},
      title = {A hyper-graph neural network trained with multi-Agent proximal policy optimization to schedule projects in engineer-to-order environments},
      year = {2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/AnasNeumann/gns}},
      commit = {main}
    }
```

## Locally try the project
1. `python -m venv ./gns_env`
2. `source ./gns_env/bin/activate`
3. `pip install --upgrade -r requirements.txt`
4. CHOOSE EITHER GNS_SOLVER, EXACT_SOLVER, INSTANCE_GENERATOR, or RESULTS_ANALYS (_see bellow for the rest_)
5. `desactivate`

## Test the instance generator
```python
bash _env.sh
python instance_generator.py --train=150 --test=50
```

## Test the exact (Google OR-Tool) solver
```python
bash _env.sh
python exact_solver.py --size=s --id=151 --path=./ --mode=test
```

## Test the GNS solver in solving mode
```python
bash _env.sh
python gns_solver.py --size=s --id=151 --train=false --mode=test --number=1 --path=./ 
```

## Test the GNS solver in training mode
```python
bash _env.sh
python gns_solver.py --train=true --mode=test --number=1 --path=./
```

## Generate exact jobs for DRAC production
* _DRAC = Digital Research Alliance of Canada_
```python
# job duration, number of CPUs, and memory used are dynamically related to the instance size (no GPU/TPU for exact jobs)
python ./jobs/exact_builder.py --account=x --parent=y --mail=x@mail.com
```

## Generate the GNS training job for DRAC production
```python
python ./jobs/gns_builder.py --account=x --parent=y --mail=x@mail.com --time=50 --memory=16 --cpu=2 --number=1
```

## Execute all jobs in DRAC production (_train GNN and solve testing instances with exact solver_)
```python
cd jobs/scripts/
bash 0_run_purge.sh
bash 1_run_all.sh train exact_s exact_m exact_l exact_xl exact_xxl exact_xxxl
```

## Locally analyze the final results with (no hard resources required): 
The main results we analyze are:
* Solution Quality: _deviation between GNS and EPSIII per instance size_
* Scalability: _computing time and memory per instance size and solution type (GNS versus EPSIII)_
* Convergence: _value loss over iterations as well as policy loss and entroy per agent and over time [GNS only]_
```python
python results_analysis.py --path=./
```
