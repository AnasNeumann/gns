# =============================================================
# =*= A REWARD TO COMPUTE LATER =*=
# =============================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

small_steps: float = 0.1
big_steps: float = 0.6
final_value: float = 0.3

class Reward:
    def __init__(self, agent_name: str, init_cost: int, init_cmax: int, makespan_old: int, makespan_new: int, last_op_old: int, last_op_new: int, cost_old: int=-1, cost_new: int=-1, a: float=-1, use_cost: bool=False):
        self.init_cost = init_cost
        self.init_cmax = init_cmax
        self.makespan_old = makespan_old
        self.makespan_new = makespan_new
        self.last_op_old = last_op_old
        self.last_op_new = last_op_new
        self.cost_old = cost_old
        self.cost_new = cost_new
        self.use_cost = use_cost
        self.a = a
        self.agent_name = agent_name

    def compute(self, final_makespan: int, final_cost: int=-1, ):
        """
            Compute the final reward
        """
        _d: float = 0.1*(self.a*self.init_cmax + (1-self.a)*self.init_cost)
        makespan_part: float = big_steps * (self.makespan_old - self.makespan_new) + small_steps * (self.last_op_old - self.last_op_new) + final_value * final_makespan
        if self.use_cost:
            cost_part: float = (big_steps+small_steps) * (self.cost_old - self.cost_new) + final_value * final_cost
            return (self.a*makespan_part + (1-self.a)*cost_part)/_d
        else:
            return (self.a*makespan_part)/_d