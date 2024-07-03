class Solution:
    def __init__(self):
        # Elements (p, e)
        self.E_start, self.E_outsourced, self.E_prod_start, self.E_validated, self.E_end = [], [], [], [], []
        # Execution of oerations (p, o, feasible r)
        self.O_uses_init_quantity, self.O_start, self.O_setup, self.O_end, self.O_executed = [], [], [], [], []
        # Relation between operations (p1, p2, o1, o2, feasible r)
        self.precedes = []
        # Design setup (p, o, r, s)
        self.D_setup = []
        # Cmax
        self.Cmax = -1

class Instance:
    def __init__(self, size, id, w_makespan, H, **kwargs):
        self.id = id
        self.size = size
        self.H = H
        self.w_makespan = w_makespan  
        
        # Global configuration
        self.M = kwargs.get('M', -1)     
        self.nb_settings = kwargs.get('nb_settings', -1)
        self.nb_HR_types = kwargs.get('nb_HR_types', -1)
        self.nb_human_resources = kwargs.get('nb_human_resources', -1)
        self.nb_production_machine_types = kwargs.get('nb_production_machine_types', -1)
        self.nb_production_machines = kwargs.get('nb_production_machines', -1)
        self.nb_material = kwargs.get('nb_material', -1)
        self.nb_ops_types = kwargs.get('nb_ops_types', -1)
        self.total_elements = kwargs.get('total_elements', -1)
        self.total_operations = kwargs.get('total_operations', -1)
        self.nb_resource_types = kwargs.get('nb_resource_types', -1)
        self.nb_resources = kwargs.get('nb_resources', -1)
        self.E_size = kwargs.get('E_size', []) #p
        self.O_size = kwargs.get('O_size', []) #p
        self.EO_size = kwargs.get('EO_size', []) #p, e

        # Resources
        self.resource_family = kwargs.get('resource_family', []) #r,rt
        self.finite_capacity = kwargs.get('finite_capacity', []) #r
        self.design_setup = kwargs.get('design_setup', []) #r, s
        self.operation_setup = kwargs.get('operation_setup', []) #r
        self.execution_time = kwargs.get('execution_time', []) #r, p, o

        # Consumable materials
        self.init_quantity = kwargs.get('init_quantity', []) #r
        self.purchase_time = kwargs.get('purchase_time', []) #r
        self.quantity_needed = kwargs.get('quantity_needed', []) #r, p, o

        # Items
        self.assembly = kwargs.get('assembly', []) #p, e1, e2
        self.direct_assembly = kwargs.get('direct_assembly', []) #p, e1, e2
        self.external = kwargs.get('external', []) #p, e
        self.outsourcing_time = kwargs.get('outsourcing_time', []) #p, e
        self.external_cost = kwargs.get('external_cost', []) #p, e 

        # Operations
        self.operation_family = kwargs.get('operation_family', []) #p, o, ot
        self.simultaneous = kwargs.get('simultaneous', []) #p, o
        self.resource_type_needed = kwargs.get('resource_type_needed', []) #p, o, rt
        self.in_hours = kwargs.get('in_hours', []) #p, o
        self.in_days = kwargs.get('in_days', []) #p, o
        self.is_design = kwargs.get('is_design', []) #p, o
        self.design_value = kwargs.get('design_value', []) #p, o, s
        self.operations_by_element = kwargs.get('operations_by_element', []) #p, e, o
        self.precedence = kwargs.get('precedence', []) #p, e, o1, o2

def get_name(i: Instance):
    return i.size+"_"+str(i.id)

def get_direct_children(i: Instance, p, e):
    children = []
    for e2 in range(i.E_size[p]):
        if i.direct_assembly[p][e][e2]:
            children.append(e2)
    return children

def get_direct_parent(i: Instance, p, e):
    for e2 in range(i.E_size[p]):
        if i.direct_assembly[p][e2][e]:
            return e2
    return -1

def get_operations_idx(i: Instance, p, e):
    start = 0
    for e2 in range(0, e):
       start = start + i.EO_size[p][e2]    
    return start, start+i.EO_size[p][e]

def require(i: Instance, p, o, r):
    for rt in range(i.nb_resource_types):
        if i.resource_family[r][rt]:
            return i.resource_type_needed[p][o][rt]
    return False

def real_time_scale(i: Instance, p, o):
    return 60*i.H if i.in_days[p][o] else 60 if i.in_hours[p][o] else 1

def get_nb_projects(i: Instance):
    return len(i.E_size)