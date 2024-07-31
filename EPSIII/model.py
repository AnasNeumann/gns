from torch_geometric.data import HeteroData
import torch
import copy

# =====================================================
# =*= SOLUTION DATA STRUCTURE =*=
# =====================================================
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
        # Cmax and objective
        self.Cmax = -1
        self.obj = []

# =====================================================
# =*= INSTANCE DATA STRUCTURE =*=
# =====================================================
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
    
    def get_name(self):
        return self.size+'_'+str(self.id)

    def get_direct_children(self, p, e):
        children = []
        for e2 in range(self.E_size[p]):
            if self.direct_assembly[p][e][e2]:
                children.append(e2)
        return children

    def get_direct_parent(self, p, e):
        for e2 in range(self.E_size[p]):
            if self.direct_assembly[p][e2][e]:
                return e2
        return -1

    def get_operations_idx(self, p, e):
        start = 0
        for e2 in range(0, e):
            start = start + self.EO_size[p][e2]    
        return start, start+self.EO_size[p][e]

    def require(self, p, o, r):
        for rt in range(self.nb_resource_types):
            if self.resource_family[r][rt]:
                return self.resource_type_needed[p][o][rt]
        return False

    def real_time_scale(self, p, o):
        return 60*self.H if self.in_days[p][o] else 60 if self.in_hours[p][o] else 1

    def get_nb_projects(self):
        return len(self.E_size)

    def project_head(self, p):
        for e in range(self.E_size[p]):
            is_head = True
            for e2 in range(self.E_size[p]):
                if e2 != e and self.assembly[p][e2][e]:
                    is_head = False
                    break
            if(is_head):
                return e
        return -1

    def last_operations(self, p, e):
        last_ops = []
        start, end = self.get_operations_idx(p, e)
        for o1 in range(start, end):
            is_last = True
            for o2 in range(start, end):
                if self.precedence[p][e][o2][o1] and o2 != o1:
                    is_last = False
                    break
            if is_last:
                last_ops.append(o1)
        return last_ops

    def required_resources(self, p, o):
        resources = []
        for r in range(self.nb_resources):
            if self.require(p, o, r):
                resources.append(r)
        return resources

    def is_same(self, p1, p2, o1, o2):
        return (p1 == p2) and (o1 == o2)

    def get_resource_familly(self, r):
        for rf in range(self.nb_resource_types):
            if self.resource_family[r][rf]:
                return rf
        return -1

    def real_time_scale(self, p, o):
        return 60*self.H if self.in_days[p][o] else 60 if self.in_hours[p][o] else 1

    def resources_by_type(self, rt):
        resources = []
        for r in range(self.nb_resources):
            if self.resource_family[r][rt]:
                resources.append(r)
        return resources

# =====================================================
# =*= HYPER-GRAPH DATA STRUCTURE =*=
# =====================================================
class State:
    def __init__(self, items, operations, resources, materials, need_for_materials, need_for_resources, operation_assembly, assembly, precedences, same_types):
        self.items = copy.deepcopy(items)
        self.operations = copy.deepcopy(operations)
        self.resources = copy.deepcopy(resources)
        self.materials = copy.deepcopy(materials)
        self.need_for_resources = copy.deepcopy(need_for_resources)
        self.need_for_materials = copy.deepcopy(need_for_materials)
        self.operation_assembly = operation_assembly
        self.assembly = assembly
        self.precedences = precedences
        self.same_types = same_types

class GraphInstance(HeteroData):
    def __init__(self):
        super()
        self.operations_g2i = []
        self.items_g2i = []
        self.resources_g2i = []
        self.materials_g2i = []
        self.current_operation_type = []
        self.current_design_value = []
        self.features = {
            'operation': {
                'physical': 0,
                'sync': 1,
                'timescale_minutes': 2,
                'timescale_hours': 3,
                'timescale_days': 4,
                'direct_successors': 5,
                'total_successors': 6,
                'remaining_time': 7,
                'remaining_resources': 8,
                'outsourced': 9,
                'available_time': 10,
                'end_time': 11
            }, 'resource': {
                'utilization_ratio': 0,
                'available_time': 1,
                'executed_operations': 2,
                'remaining_operations': 3,
                'similar_resources': 4
            }, 'material': {
                'quantity': 0,
                'time_before_arrival': 1,
                'remaining_demand': 2
            }, 'item': {
                'head': 0,
                'outsourced_yet': 1,
                'outsourced': 2,
                'outsourcing_cost': 3,
                'outsourcing_time': 4,
                'deadline': 5,
                'remaining_physical_time': 5,
                'remaining_design_time': 6,
                'parents': 7,
                'children': 8,
                'parents_physical_time': 9,
                'children_design_time': 10,
                'start_time': 11,
                'end_time': 12
            }, 'need_for_resources': {
                'status': 0,
                'basic_processing_time': 1,
                'current_processing_time': 2,
                'start_time': 3,
                'end_time': 4
            }, 'need_for_materials': {
                'status': 0,
                'execution_time': 1,
                'quantity_needed': 2
            }
        }

    def build_from_instance(self, instance: Instance):
        # TODO translate an instance into a graph structure
        pass

    def add_node(self, type, features):
        self[type].x = torch.cat([self[type].x, features], dim=0) if type in self.node_types else features

    def add_edge(self, node_1, relation, node_2, features):
        self[node_1, relation, node_2].edge_index = torch.cat([self[node_1, relation, node_2].edge_index, features], dim=1) if (node_1, relation, node_2) in self.edge_types else features
    
    def precedences(self):
        return self['operation', 'precedes', 'operation'].edge_index
    
    def item_assembly(self):
        return self['item', 'parent', 'item'].edge_index
    
    def operation_assembly(self):
        return self['item', 'has', 'operation'].edge_index
    
    def need_for_resources(self):
        return self['operation', 'needs_res', 'resource'].edge_index
    
    def need_for_materials(self):
        return self['operation', 'needs_mat', 'material'].edge_index
    
    def same_types(self):
        return self['resource', 'same', 'resource'].edge_index
    
    def operations(self):
        return self['operation'].x
    
    def items(self):
        return self['item'].x
    
    def resources(self):
        return self['resource'].x
    
    def materials(self):
        return self['material'].x

    def operation(self, id, feature):
        return self['operation'].x[id][self.features['operation'][feature]].item()

    def material(self, id, feature):
        return self['material'].x[id][self.features['material'][feature]].item()
    
    def resource(self, id, feature):
        return self['resource'].x[id][self.features['resource'][feature]].item()
    
    def item(self, id, feature):
        return self['item'].x[id][self.features['item'][feature]].item()
    
    def update_operation(self, id, updates):
        for feature, value in updates:
            self['operation'].x[id][self.features['operation'][feature]] = value
        
    def update_resource(self, id, updates):
        for feature, value in updates:
            self['resource'].x[id][self.features['resource'][feature]] = value
    
    def update_material(self, id, updates):
        for feature, value in updates:
            self['material'].x[id][self.features['material'][feature]] = value
    
    def update_item(self, id, updates):
        for feature, value in updates:
            self['item'].x[id][self.features['item'][feature]] = value

    def update_operation(self, id, updates):
        for feature, value in updates:
            self['operation'].x[id][self.features['operation'][feature]] = value

    def update_need_for_material(self, operation_id, material_id, updates):
        key = (operation_id, 'needs_mat', material_id)
        idx = (self[key].edge_index[0] == operation_id) & (self[key].edge_index[1] == material_id)
        for feature, value in updates:
            self[key].edge_attr[idx, self.features['need_for_materials'][feature]] = value

    def update_need_for_resources(self, operation_id, resource_id, updates):
        key = (operation_id, 'needs_res', resource_id)
        idx = (self[key].edge_index[0] == operation_id) & (self[key].edge_index[1] == resource_id)
        for feature, value in updates:
            self[key].edge_attr[idx, self.features['need_for_resources'][feature]] = value

    def to_state(self):
        state = State(self.items, 
                      self.operations, 
                      self.resources, 
                      self.materials, 
                      self.need_for_materials, 
                      self.need_for_resources, 
                      self.operation_assembly, 
                      self.item_assembly, 
                      self.precedences, 
                      self.same_types)
        return state
    
    def feasible_actions(self):
        # TODO return all feasible actions
        actions = []
        return actions