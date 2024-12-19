from model.instance import Instance
from tools.common import init_several_1D, init_2D, init_several_2D, init_3D
import random

# ====================================================================
# =*= EXACT (Google OR-Tool) SOLUTION DATA STRUCTURE =*=
# ====================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

class Solution:
    def __init__(self):
        self.E_start, self.E_outsourced, self.E_prod_start, self.E_validated, self.E_end = [], [], [], [], [] # Elements (p, e)
        self.O_uses_init_quantity, self.O_start, self.O_setup, self.O_end, self.O_executed = [], [], [], [], [] # Execution of operations (p, o, feasible r)
        self.precedes = [] # Relation between operations (p1, p2, o1, o2, feasible r)
        self.D_setup = [] # Design setup (p, o, r, s)
        self.Cmax = -1 # Cmax and objective
        self.obj = []

class HeuristicSolution():
    def __init__(self):
        self.sequences = [] # Sequences of each resource rt = (p,o)
        self.selection = [] # Selection of resourses for each operations and feasible resource type p, o, rt = r
        self.E_start, self.outsourced, self.E_prod_start, self.E_validated, self.E_end = [] # Elements (p, e)
        self.O_start, self.O_end = [], [] # Execution of operations (p, o, rt)
        self.total_cost = 0
        self.Cmax = -1

    def random_start_from_instance(self, i: Instance):
        nb_projects = i.get_nb_projects()
        elts_per_project = range(i.E_size[0])
        self.sequences = [[(p,o) for p, o in i.operations_by_resource_type(rt)] for rt in i.loop_rts()]
        for rt in i.loop_rts():
            random.shuffle(self.sequences[rt])
        self.selection = [[[random.choice(i.resources_by_type(rt)) for rt in i.required_rt(p,o)] for o in i.loop_operations(p)] for p in i.loop_projects()]
        self.E_start = [[-1 for _ in elts_per_project] for p in i.loop_projects()]
        self.outsourced = [[False for _ in elts_per_project] for _ in i.loop_projects()]
        self.E_prod_start = [[-1 for _ in elts_per_project] for _ in i.loop_projects()]
        self.E_end = [[-1 for _ in elts_per_project] for _ in i.loop_projects()]
        self.E_validated = [[-1 for _ in elts_per_project] for _ in i.loop_projects()]
        self.O_start, self.O_end = init_several_1D(nb_projects, None, 2)
        for p in i.loop_projects():
            nb_ops = i.O_size[p]
            self.O_start[p], self.O_end[p] = init_several_2D(nb_ops, i.nb_resource_types, -1, 2)

    def display(self, i:Instance):
        # TODO
        pass

    def simulate(self, i: Instance):
        # TODO
        pass