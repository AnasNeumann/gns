import copy
import torch
from torch_geometric.data import HeteroData
from torch.nn import Sequential, Linear, ELU, Tanh, Parameter, LeakyReLU, Module
import torch.nn.functional as F

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

    def add_node(self, type, features):
        self[type].x = torch.cat([self[type].x, features], dim=0) if type in self.node_types else features

    def add_edge(self, node_1, relation, node_2, idx, features):
        self[node_1, relation, node_2].edge_index = torch.cat([self[node_1, relation, node_2].edge_index, idx], dim=1) if (node_1, relation, node_2) in self.edge_types else idx
        self[node_1, relation, node_2].edge_attr = torch.cat([self[node_1, relation, node_2].edge_attr, features], dim=1) if (node_1, relation, node_2) in self.edge_types else features
    
    def precedences(self):
        return self['operation', 'precedes', 'operation']
    
    def item_assembly(self):
        return self['item', 'parent', 'item']
    
    def operation_assembly(self):
        return self['item', 'has', 'operation']
    
    def need_for_resources(self):
        return self['operation', 'needs_res', 'resource']
    
    def need_for_materials(self):
        return self['operation', 'needs_mat', 'material']
    
    def same_types(self):
        return self['resource', 'same', 'resource']
    
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

    def update_need_for_resource(self, operation_id, resource_id, updates):
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

# =====================================================
# =*= GRAPH ATTENTION NEURAL NETWORK (GaNN) =*=
# =====================================================

class MaterialEmbedding(Module):
    def __init__(self, material_dimension, operation_dimension, embedding_dimension):
        super(MaterialEmbedding, self).__init__()
        self.material_transform = Linear(material_dimension, embedding_dimension, bias=False)
        self.att_self_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.operation_transform = Linear(operation_dimension, embedding_dimension, bias=False)
        self.att_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.leaky_relu = LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.material_transform.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.operation_transform.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_coef.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_self_coef.data, gain=1.414)

    def forward(self, materials, operations, need_for_materials):
        materials = self.material_transform(materials)
        self_attention = self.leaky_relu(torch.matmul(torch.cat([materials, materials], dim=-1), self.att_self_coef))
        
        ops_by_edges = self.operation_transform(torch.cat([operations[need_for_materials.edge_index[0]], need_for_materials.edge_attr], dim=-1))
        mat_by_edges = materials[need_for_materials.edge_index[1]]
        cross_attention = self.leaky_relu(torch.matmul(torch.cat([mat_by_edges, ops_by_edges], dim=-1), self.att_coef))

        normalizer = F.softmax(torch.cat([self_attention, cross_attention], dim=0), dim=0)
        norm_self_attention = normalizer[:self_attention.size(0)]
        norm_cross_attention = normalizer[self_attention.size(0):]

        weighted_ops_by_edges = norm_cross_attention * ops_by_edges
        sum_ops_by_edges = torch.zeros_like(materials, device=materials.device)
        sum_ops_by_edges.index_add_(0, need_for_materials.edge_index[1], weighted_ops_by_edges)
        embedding = F.elu(norm_self_attention * materials + sum_ops_by_edges)
        return embedding

class ResourceEmbedding(Module):
    def __init__(self, resource_dimension, operation_dimension, embedding_dimension):
        super(MaterialEmbedding, self).__init__()
        self.self_transform = Linear(resource_dimension, embedding_dimension, bias=False)
        self.resource_transform = Linear(resource_dimension, embedding_dimension, bias=False)
        self.operation_transform = Linear(operation_dimension, embedding_dimension, bias=False)
        self.att_operation_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.att_resource_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.att_self_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.leaky_relu = LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.self_transform.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.resource_transform.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.operation_transform.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_operation_coef.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_resource_coef.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_self_coef.data, gain=1.414)

    def forward(self, resources, operations, need_for_resources, same_types):
        self_resources = self.self_transform(resources) 
        self_attention = self.leaky_relu(torch.matmul(torch.cat([self_resources, self_resources], dim=-1), self.att_self_coef))

        ops_by_need_edges = self.operation_transform(torch.cat([operations[need_for_resources.edge_index[0]], need_for_resources.edge_attr], dim=-1))
        res_by_need_edges = self_resources[need_for_resources.edge_index[1]]
        operations_cross_attention = self.leaky_relu(torch.matmul(torch.cat([res_by_need_edges, ops_by_need_edges], dim=-1), self.att_operation_coef))
        
        res1_by_same_edges = self.resource_transform(resources[same_types.edge_index[0]])
        res2_by_same_edges = self_resources[same_types.edge_index[1]]
        resources_cross_attention = self.leaky_relu(torch.matmul(torch.cat([res2_by_same_edges, res1_by_same_edges], dim=-1), self.att_resource_coef))
        
        normalizer = F.softmax(torch.cat([self_attention, operations_cross_attention, resources_cross_attention], dim=0), dim=0)
        norm_self_attention = normalizer[:self_attention.size(0)+operations_cross_attention.size(0)]
        norm_operations_cross_attention = normalizer[self_attention.size(0):operations_cross_attention.size(0)]
        norm_resources_cross_attention = normalizer[self_attention.size(0)+operations_cross_attention.size(0):]

        weighted_ops_by_edges = norm_operations_cross_attention * ops_by_need_edges
        sum_ops_by_edges = torch.zeros_like(resources, device=resources.device)
        sum_ops_by_edges.index_add_(0, need_for_resources.edge_index[1], weighted_ops_by_edges)

        weighted_res_by_edges = norm_resources_cross_attention * res1_by_same_edges
        sum_res_by_edges = torch.zeros_like(resources, device=resources.device)
        sum_res_by_edges.index_add_(0, same_types.edge_index[1], weighted_res_by_edges)
        
        embedding = F.elu(norm_self_attention * resources + sum_ops_by_edges + sum_res_by_edges)
        return embedding
    
class ItemLayer(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ItemLayer, self).__init__()
        self.embedding_size = out_channels
        self.mlp_combined = Sequential(
            Linear(4 * out_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_predecessor = Sequential(
            Linear(in_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_successor = Sequential(
            Linear(in_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_resources = Sequential(
            Linear(out_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_same = Sequential(
            Linear(in_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, operations, resources, requirement_edges, preds, succs):
        ops_idx_by_edges = requirement_edges[0]
        res_embeddings_by_edges = resources[requirement_edges[1]]
        agg_machine_embeddings = torch.zeros((operations.size(0), resources.size(1)), device=operations.device)
        agg_machine_embeddings.index_add_(0, ops_idx_by_edges, res_embeddings_by_edges)
        predecessors = self.mlp_predecessor(operations[preds[1:-1]])
        successors = self.mlp_successor(operations[succs[1:-1]])
        same_embeddings = self.mlp_same(operations[1:-1])
        agg_machine_embeddings = self.mlp_resources(agg_machine_embeddings[1:-1])
        embedding = torch.zeros((operations.shape[0], self.embedding_size), device=operations.device)
        embedding[1:-1] = self.mlp_combined(torch.cat([predecessors, successors, agg_machine_embeddings, same_embeddings], dim=-1))
        return embedding
    
class OperationLayer(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(OperationLayer, self).__init__()
        self.embedding_size = out_channels
        self.mlp_combined = Sequential(
            Linear(4 * out_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_predecessor = Sequential(
            Linear(in_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_successor = Sequential(
            Linear(in_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_resources = Sequential(
            Linear(out_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_same = Sequential(
            Linear(in_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, operations, resources, requirement_edges, preds, succs):
        ops_idx_by_edges = requirement_edges[0]
        res_embeddings_by_edges = resources[requirement_edges[1]]
        agg_machine_embeddings = torch.zeros((operations.size(0), resources.size(1)), device=operations.device)
        agg_machine_embeddings.index_add_(0, ops_idx_by_edges, res_embeddings_by_edges)
        predecessors = self.mlp_predecessor(operations[preds[1:-1]])
        successors = self.mlp_successor(operations[succs[1:-1]])
        same_embeddings = self.mlp_same(operations[1:-1])
        agg_machine_embeddings = self.mlp_resources(agg_machine_embeddings[1:-1])
        embedding = torch.zeros((operations.shape[0], self.embedding_size), device=operations.device)
        embedding[1:-1] = self.mlp_combined(torch.cat([predecessors, successors, agg_machine_embeddings, same_embeddings], dim=-1))
        return embedding