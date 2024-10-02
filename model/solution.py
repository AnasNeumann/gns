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