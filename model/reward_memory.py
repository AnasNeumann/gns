# =============================================================================================
# =*= A REPLAY MEMORY SYSTEMS TO SAVE DECISIONS MADE WHILE DESTINGUISHING BETWEEN INSTANCES =*=
# =============================================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

w_final: float = 0.65
standardization: float = 1.0

class Decision:
    """
        One decision
    """
    def __init__(self, type: int, agent_name: str, target: int, value: int, end_old: int, end_new: int, cost_old: int=-1, cost_new: int=-1, parent=None, use_cost: bool=False):
        self.type: int = type
        self.target: int = target
        self.value: int = value
        self.end_old: int = end_old
        self.end_new: int = end_new
        self.cost_old: int = cost_old
        self.agent_name: str = agent_name
        self.cost_new: int = cost_new
        self.use_cost: int = use_cost
        self.next_decisions: list[Decision] = []
        self.parent: Decision = parent
        self.reward: float = -1
        if parent is not None:
            self.parent.next_decisions.append(self)

    def same(self, d) -> bool:
        """
            Check if two decisions are the same
        """
        d: Decision
        return d.parent == self.parent and d.type==self.type and d.target==self.target and d.value==self.value

    def compute_reward(self, a: float, init_cmax: int, init_cost: int, final_makespan: int, final_cost: int=-1) -> float:
        """
            Compute the final reward
        """
        _d: float = standardization*(a*init_cmax + (1-a)*init_cost)
        makespan_part: float =  (1.0-w_final) * (self.end_new - self.end_old) + w_final * final_makespan
        if self.use_cost:
            cost_part: float = (1.0-w_final) * (self.cost_new - self.cost_old) + w_final * final_cost
            self.reward = -1.0 * (a*makespan_part + (1-a)*cost_part)/_d
        else:
            self.reward = -(a*makespan_part)/_d
        return self.reward

class Memory:
    """
        The memory of one specific instance (both pre-training and fine-tuning)
    """
    def __init__(self, instance_id: int):
        self.instance_id: int = instance_id
        self.decisions: list[Decision] = []

    def compute_all_rewards(self, decision: Decision, a: float, init_cmax: int, init_cost: int, final_makespan: int, final_cost: int=-1) -> None:
        """
            Compute all rewards
        """
        decision.compute_reward(a=a, final_cost=final_cost, final_makespan=final_makespan, init_cmax=init_cmax, init_cost=init_cost)
        for _next in decision.next_decisions:
            self.compute_all_rewards(decision=_next, a=a, final_cost=final_cost, final_makespan=final_makespan, init_cmax=init_cmax, init_cost=init_cost)

    def add_or_update_decision(self, decision: Decision, a: float, init_cmax: int, init_cost: int, final_makespan: int, final_cost: int=-1, need_rewards: bool=True) -> Decision:
        """
            Add decision in the memory or update reward if already exist
        """
        if need_rewards:
            self.compute_all_rewards(decision=decision, a=a, final_cost=final_cost, final_makespan=final_makespan, init_cmax=init_cmax, init_cost=init_cost)
        if decision.parent is None:
            _found: bool = False
            for _other_first in self.decisions:
                if _other_first.same(decision):
                    _found = True
                    _other_first.reward = min(_other_first.reward, decision.reward)
                    for _next_decision in decision.next_decisions:
                        _next_decision.parent = _other_first
                        self.add_or_update_decision(decision=_next_decision, 
                                                    a=a, 
                                                    final_cost=final_cost, 
                                                    final_makespan=final_makespan, 
                                                    init_cmax=init_cmax, 
                                                    init_cost=init_cost,
                                                    need_rewards=False)
                    return _other_first
            if not _found:
                self.decisions.append(decision)
                return decision
        else:
            _found: bool = False
            for _next in decision.parent.next_decisions:
                if _next.same(decision):
                    _found = True
                    _next.reward = min(_next.reward, decision.reward)
                    for _next_decision in decision.next_decisions:
                        _next_decision.parent = _next
                        self.add_or_update_decision(decision=_next_decision, 
                                                    a=a, 
                                                    final_cost=final_cost, 
                                                    final_makespan=final_makespan, 
                                                    init_cmax=init_cmax, 
                                                    init_cost=init_cost,
                                                    need_rewards=False)
                    return next
            if not _found:
                decision.parent.next_decisions.append(decision)
                return decision

class Memories:
    """
        The memory for several instances (pre-training time only)
    """
    def __init__(self):
        self.instances: list[Memory] = []
    
    def add_instance_if_new(self, instance_id: int) -> Memory:
        """
            Add a new instance if ID is not present yet
        """
        for memory in self.instances:
            if memory.instance_id == instance_id:
                return memory
        new_memory: Memory = Memory(instance_id=instance_id)
        self.instances.append(new_memory)
        return new_memory
