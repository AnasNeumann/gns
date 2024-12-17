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
        self.O_uses_init_quantity, self.O_start, self.O_setup, self.O_end, self.O_executed = [], [], [], [], [] # Execution of oerations (p, o, feasible r)
        self.precedes = [] # Relation between operations (p1, p2, o1, o2, feasible r)
        self.D_setup = [] # Design setup (p, o, r, s)
        self.Cmax = -1 # Cmax and objective
        self.obj = []

class HeuristicSolution(Solution):
    def __init__(self):
        Solution.__init__(self)
        self.sequences = [] # Sequences of each resource rt = (p,o)
        self.selection = [] # Selection of resourses for each operations and feasible resource type p, o, rt = r

    def random_start_from_instance(self, i: Instance):
        nb_projects = i.get_nb_projects()
        elts_per_project = range(i.E_size[0])
        self.sequences = [[(p,o) for p, o in i.operations_by_resource_type(rt)] for rt in i.loop_rts()]
        for rt in i.loop_rts():
            random.shuffle(self.sequences[rt])
        self.selection = [[[random.choice(i.resources_by_type(rt)) for rt in i.required_rt(p,o)] for o in i.loop_operations(p)] for p in i.loop_projects()]
        self.E_start = [[0 for e in elts_per_project] for p in i.loop_projects()]
        self.E_outsourced = [[False for e in elts_per_project] for p in i.loop_projects()]
        self.E_prod_start = [[0 for e in elts_per_project] for p in i.loop_projects()]
        self.E_end = [[0 for e in elts_per_project] for p in i.loop_projects()]
        self.E_validated = [[0 for e in elts_per_project] for p in i.loop_projects()]
        self.O_uses_init_quantity, self.O_start, self.O_setup, self.O_end, self.O_executed, self.D_setup = init_several_1D(nb_projects, None, 6)
        self.precedes = init_2D(nb_projects, nb_projects, None)
        self.Cmax = 0
        for p in i.loop_projects():
            nb_ops = i.O_size[p]
            self.O_uses_init_quantity[p], self.O_setup[p], self.O_executed[p] = init_several_2D(nb_ops, i.nb_resources, False, 3)
            self.O_start[p], self.O_end[p] = init_several_2D(nb_ops, i.nb_resources, 0, 2)
            self.D_setup[p] = init_3D(nb_ops, i.nb_resources, i.nb_settings, False)
            for p2 in i.loop_projects():
                nb_ops2 = i.O_size[p2]
                self.precedes[p][p2] = init_3D(nb_ops, nb_ops2, i.nb_resources, False)