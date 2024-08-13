import copy
import torch
from torch_geometric.data import HeteroData
from torch.nn import Sequential, Linear, ELU, Tanh, Parameter, LeakyReLU, Module, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch import Tensor
from common import features2tensor, id2tensor, init_3D

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
        self.resource_family = kwargs.get('resource_family', []) #r,rt (boolean)
        self.finite_capacity = kwargs.get('finite_capacity', []) #r (boolean)
        self.design_setup = kwargs.get('design_setup', []) #r, s
        self.operation_setup = kwargs.get('operation_setup', []) #r
        self.execution_time = kwargs.get('execution_time', []) #r, p, o

        # Consumable materials
        self.init_quantity = kwargs.get('init_quantity', []) #r
        self.purchase_time = kwargs.get('purchase_time', []) #r
        self.quantity_needed = kwargs.get('quantity_needed', []) #r, p, o

        # Items
        self.assembly = kwargs.get('assembly', []) #p, e1, e2 (boolean)
        self.direct_assembly = kwargs.get('direct_assembly', []) #p, e1, e2 (boolean)
        self.external = kwargs.get('external', []) #p, e (boolean)
        self.outsourcing_time = kwargs.get('outsourcing_time', []) #p, e
        self.external_cost = kwargs.get('external_cost', []) #p, e 

        # Operations
        self.operation_family = kwargs.get('operation_family', []) #p, o, ot (boolean)
        self.simultaneous = kwargs.get('simultaneous', []) #p, o (boolean)
        self.resource_type_needed = kwargs.get('resource_type_needed', []) #p, o, rt (boolean)
        self.in_hours = kwargs.get('in_hours', []) #p, o (boolean)
        self.in_days = kwargs.get('in_days', []) #p, o (boolean)
        self.is_design = kwargs.get('is_design', []) #p, o (boolean)
        self.design_value = kwargs.get('design_value', []) #p, o, s
        self.operations_by_element = kwargs.get('operations_by_element', []) #p, e, o (boolean)
        self.precedence = kwargs.get('precedence', []) #p, e, o1, o2 (boolean)
    
    def get_name(self):
        return self.size+'_'+str(self.id)

    def get_children(self, p, e, direct=True):
        data = self.direct_assembly if direct else self.assembly
        children = []
        for e2 in range(self.E_size[p]):
            if data[p][e][e2]:
                children.append(e2)
        return children

    def get_direct_parent(self, p, e):
        for e2 in range(self.E_size[p]):
            if self.direct_assembly[p][e2][e]:
                return e2
        return -1

    def get_ancestors(self, p, e):
        ancestors = []
        for e2 in range(self.E_size[p]):
            if self.assembly[p][e][e2]:
                ancestors.append(e2)
        return ancestors
    
    def get_operations_idx(self, p, e):
        start = 0
        for e2 in range(0, e):
            start = start + self.EO_size[p][e2]    
        return start, start+self.EO_size[p][e]
    
    def operation_time(self, p, o):
        time = 0
        rts = self.required_rt(p,o)
        for rt in rts:
            time_rt = 0
            for r in self.resources_by_type(rt):
                if self.finite_capacity[r]:
                    time_rt = max(time_rt, self.execution_time[r, p, o])
            time += time_rt
        return time

    def item_processing_time(self, p, e):
        start, end = self.get_operations_idx(p,e)
        design_time = 0
        physical_time = 0
        for o in range(start, end):
            if self.is_design[p][o]:
                design_time += self.operation_time(p,o)
            else:
                physical_time += self.operation_time(p,o)
        return design_time, physical_time

    def require(self, p, o, r):
        for rt in range(self.nb_resource_types):
            if self.resource_family[r][rt]:
                return self.resource_type_needed[p][o][rt]
        return False
    
    def required_rt(self, p, o):
        rts = []
        for rt in range(self.nb_resource_types):
            if self.resource_type_needed[p][o][rt]:
                rts.append(rt)
        return rts

    def real_time_scale(self, p, o):
        return 60*self.H if self.in_days[p][o] else 60 if self.in_hours[p][o] else 1

    def get_nb_projects(self):
        return len(self.E_size)
    
    def operations_by_resource_type(self, rt):
        operations = []
        for p in range(self.get_nb_projects()):
            for o in range(self.O_size[p]):
                if self.resource_type_needed[p][o][rt]:
                    operations.append((p, o))
        return operations

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
    def __init__(self, items, operations, resources, materials, need_for_materials, need_for_resources, operation_assembly, item_assembly, precedences, same_types):
        self.items = copy.deepcopy(items)
        self.operations = copy.deepcopy(operations)
        self.resources = copy.deepcopy(resources)
        self.materials = copy.deepcopy(materials)
        self.need_for_resources = copy.deepcopy(need_for_resources)
        self.need_for_materials = copy.deepcopy(need_for_materials)
        self.operation_assembly = operation_assembly
        self.item_assembly = item_assembly
        self.precedences = precedences
        self.same_types = same_types

class FeatureConfiguration:
    def __init__(self):
        self.operation = {
            'design': 0,
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
        }
        self.resource = {
            'utilization_ratio': 0,
            'available_time': 1,
            'executed_operations': 2,
            'remaining_operations': 3,
            'similar_resources': 4
        }
        self.material = {
            'quantity': 0,
            'time_before_arrival': 1,
            'remaining_demand': 2
        }
        self.item = {
            'head': 0,
            'external': 1,
            'outsourced': 2,
            'outsourcing_cost': 3,
            'outsourcing_time': 4,
            'remaining_physical_time': 5,
            'remaining_design_time': 6,
            'parents': 7,
            'children': 8,
            'parents_physical_time': 9,
            'children_design_time': 10,
            'start_time': 11,
            'end_time': 12
        }
        self.need_for_resources = {
            'status': 0,
            'basic_processing_time': 1,
            'current_processing_time': 2,
            'start_time': 3,
            'end_time': 4
        }
        self.need_for_materials = {
            'status': 0,
            'execution_time': 1,
            'quantity_needed': 2
        }

class GraphInstance(HeteroData):
    def __init__(self):
        super()
        self.operations_g2i = []
        self.items_g2i = []
        self.resources_g2i = []
        self.materials_g2i = []

        self.operations_i2g = []
        self.items_i2g = []
        self.resources_i2g = []
        self.materials_i2g = []
        self.current_operation_type = []
        self.current_design_value = []
        self.project_heads = []
        self.features = FeatureConfiguration()

    def add_node(self, type, features: Tensor):
        self[type].x = torch.cat([self[type].x, features], dim=0) if type in self.node_types else features

    def add_operation(self, p, o, *args):
        self.operations_g2i.append((p, o))
        self.add_node('operation', features2tensor(args))
        return len(self.operations_g2i)-1

    def add_item(self, p, i, *args):
        self.items_g2i.append((p, i))
        self.add_node('item', features2tensor(args))
        id = len(self.items_g2i)-1
        if args[self.features.item['head']] == 1:
            self.project_heads.append(id)
        return id

    def add_material(self, m, *args):
        self.materials_g2i.append(m)
        self.add_node('material', features2tensor(args))
        return len(self.materials_g2i)-1

    def add_resource(self, r, nb_settings, *args):
        self.resources_g2i.append(r)
        self.current_design_value.append([-1 for _ in range(nb_settings)])
        self.current_operation_type.append(-1)
        self.add_node('resource', features2tensor(args))
        return len(self.resources_g2i)-1

    def add_edge_no_features(self, node_1, relation, node_2, idx):
        self[node_1, relation, node_2].edge_index = torch.cat([self[node_1, relation, node_2].edge_index, idx], dim=1) if (node_1, relation, node_2) in self.edge_types else idx

    def add_same_types(self, res_1, res_2):
        self.add_edge_no_features('resource', 'same', 'resource', id2tensor(res_1, res_2))
        self.add_edge_no_features('resource', 'same', 'resource', id2tensor(res_2, res_1))

    def add_item_assembly(self, parent_id, child_id):
        self.add_edge_no_features('item', 'parent', 'item', id2tensor(parent_id, child_id))

    def add_operation_assembly(self, item_id, operation_id):
        self.add_edge_no_features('item', 'has', 'operation', id2tensor(item_id, operation_id))

    def add_precedence(self, prec_id, succ_id):
        self.add_edge_no_features('operation', 'precedes', 'operation', id2tensor(prec_id, succ_id))

    def add_edge_with_features(self, node_1, relation, node_2, idx, features: Tensor):
        self[node_1, relation, node_2].edge_index = torch.cat([self[node_1, relation, node_2].edge_index, idx], dim=1) if (node_1, relation, node_2) in self.edge_types else idx
        self[node_1, relation, node_2].edge_attr = torch.cat([self[node_1, relation, node_2].edge_attr, features], dim=1) if (node_1, relation, node_2) in self.edge_types else features
    
    def add_need_for_materials(self, operation_id, material_id, features):
        self.add_edge_no_features('operation', 'needs_mat', 'material', id2tensor(operation_id, material_id), features2tensor(features))

    def add_need_for_resources(self, operation_id, resource_id, features):
        self.add_edge_no_features('operation', 'needs_res', 'resource', id2tensor(operation_id, resource_id), features2tensor(features))

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
        return self['operation'].x[id][self.features.operation[feature]].item()

    def material(self, id, feature):
        return self['material'].x[id][self.features.material[feature]].item()
    
    def resource(self, id, feature):
        return self['resource'].x[id][self.features.resource[feature]].item()
    
    def item(self, id, feature):
        return self['item'].x[id][self.features.item[feature]].item()
    
    def update_operation(self, id, updates):
        for feature, value in updates:
            self['operation'].x[id][self.features.operation[feature]] = value
        
    def update_resource(self, id, updates):
        for feature, value in updates:
            self['resource'].x[id][self.features.resource[feature]] = value
    
    def update_material(self, id, updates):
        for feature, value in updates:
            self['material'].x[id][self.features.material[feature]] = value
    
    def update_item(self, id, updates):
        for feature, value in updates:
            self['item'].x[id][self.features.item[feature]] = value

    def update_operation(self, id, updates):
        for feature, value in updates:
            self['operation'].x[id][self.features.operation[feature]] = value

    def update_need_for_material(self, operation_id, material_id, updates):
        key = (operation_id, 'needs_mat', material_id)
        idx = (self[key].edge_index[0] == operation_id) & (self[key].edge_index[1] == material_id)
        for feature, value in updates:
            self[key].edge_attr[idx, self.features.need_for_materials[feature]] = value

    def update_need_for_resource(self, operation_id, resource_id, updates):
        key = (operation_id, 'needs_res', resource_id)
        idx = (self[key].edge_index[0] == operation_id) & (self[key].edge_index[1] == resource_id)
        for feature, value in updates:
            self[key].edge_attr[idx, self.features.need_for_resources[feature]] = value

    def parents(self):
        adj = to_dense_adj(self.item_assembly().edge_index)[0]
        nb_items = self.items().size(0)
        parents = torch.zeros(nb_items, dtype=torch.long)
        for i in range(nb_items):
            parents[i] = adj[:,i].nonzero(as_tuple=True)[0]
        return parents

    def related_items(self):
        adj = to_dense_adj(self.operation_assembly().edge_index)[0]
        nb_ops = self.operations().size(0)
        r_items = torch.zeros(nb_ops, dtype=torch.long)
        for i in range(nb_ops):
            r_items[i] = adj[:,i].nonzero(as_tuple=True)[0]
        return r_items

    def build_i2g_2D(self, g2i):
        nb_project = max(val[0] for val in g2i) + 1
        i2g = [[] for _ in range(nb_project)]
        for position, (project, op_or_item) in enumerate(g2i):
            while len(i2g[project]) <= op_or_item:
                i2g[project].append(-1)
            i2g[project][op_or_item] = position
        return i2g
    
    def build_i2g_1D(self, g2i):
        i2g = [0] * len(g2i)
        for id in range(len(g2i)):
            i2g[g2i[id]] = id
        return i2g

    def get_direct_children(self, instance, item_id):
        p, e = self.items_g2i[item_id]
        children = []
        for child in instance.get_children(p,e,direct=True):
            children.append(self.items_i2g[p][child])
        return children

    def to_state(self):
        state = State(self.items(), 
                      self.operations(), 
                      self.resources(), 
                      self.materials(), 
                      self.need_for_materials(), 
                      self.need_for_resources(), 
                      self.operation_assembly(), 
                      self.item_assembly(), 
                      self.precedences(), 
                      self.same_types())
        return state

# =====================================================
# =*= GRAPH ATTENTION NEURAL NETWORK (GaNN) =*=
# =====================================================

OUTSOURCING = "outsourcing"
SCHEDULING = "scheduling"
MATERIAL_USE = "material_use"

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
    def __init__(self, operation_dimension, item_dimension, hidden_channels, out_channels):
        super(ItemLayer, self).__init__()
        self.embedding_size = out_channels
        self.mlp_combined = Sequential(
            Linear(4 * out_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_operations = Sequential(
            Linear(operation_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_parent = Sequential(
            Linear(item_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_children = Sequential(
            Linear(item_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_self = Sequential(
            Linear(item_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, items, parents, operations, item_assembly, operation_assembly):
        self_embeddings = self.mlp_self(items)
        parent_embeddings = self.mlp_parent(parents)

        parent_idx_by_edge = item_assembly.edge_index[0]
        children_by_edge = items[item_assembly.edge_index[1]]
        agg_children_embeddings = torch.zeros((items.size(0), items.size(1)), device=items.device)
        agg_children_embeddings.index_add_(0, parent_idx_by_edge, children_by_edge) 
        agg_children_embeddings = self.mlp_children(agg_children_embeddings)

        item_idx_by_edges = operation_assembly.edge_index[0]
        operations_by_edges = operations[operation_assembly.edge_index[1]]
        agg_ops_embeddings = torch.zeros((items.size(0), operations.size(1)), device=items.device)
        agg_ops_embeddings.index_add_(0, item_idx_by_edges, operations_by_edges) 
        agg_ops_embeddings = self.mlp_operations(agg_ops_embeddings)
        
        embedding = torch.zeros((items.shape[0], self.embedding_size), device=items.device)
        embedding[1:-1] = self.mlp_combined(torch.cat([parent_embeddings, agg_children_embeddings, agg_ops_embeddings, self_embeddings], dim=-1))
        return embedding
    
class OperationLayer(Module):
    def __init__(self, operation_dimension, item_dimension, resources_dimension, material_dimension, hidden_channels, out_channels):
        super(OperationLayer, self).__init__()
        self.embedding_size = out_channels
        self.mlp_combined = Sequential(
            Linear(6 * out_channels, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_items = Sequential(
            Linear(item_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_predecessors = Sequential(
            Linear(operation_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_successors = Sequential(
            Linear(operation_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_resources = Sequential(
            Linear(resources_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_materials = Sequential(
            Linear(material_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )
        self.mlp_self = Sequential(
            Linear(operation_dimension, hidden_channels), ELU(),
            Linear(hidden_channels, hidden_channels), ELU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, operations, related_items, materials, resources, need_for_resources, need_for_materials, precedences):
        self_embeddings = self.mlp_self(operations)
        item_embeddings = self.mlp_items(related_items)

        operations_idx_by_mat_edge = need_for_materials.edge_index[0]
        materials_by_edge = materials[need_for_materials.edge_index[1]]
        agg_materials_embeddings = torch.zeros((operations.size(0), materials.size(1)), device=operations.device)
        agg_materials_embeddings.index_add_(0, operations_idx_by_mat_edge, materials_by_edge) 
        agg_materials_embeddings = self.mlp_materials(agg_materials_embeddings)

        operations_idx_by_res_edge = need_for_resources.edge_index[0]
        resources_by_edge = resources[need_for_resources.edge_index[1]]
        agg_resources_embeddings = torch.zeros((operations.size(0), resources.size(1)), device=operations.device)
        agg_resources_embeddings.index_add_(0, operations_idx_by_res_edge, resources_by_edge) 
        agg_resources_embeddings = self.mlp_resources(agg_resources_embeddings)

        operations_idx_by_pred_edge = precedences.edge_index[0]
        preds_by_edge = resources[precedences.edge_index[1]]
        agg_preds_embeddings = torch.zeros((operations.size(0), operations.size(1)), device=operations.device)
        agg_preds_embeddings.index_add_(0, operations_idx_by_pred_edge, preds_by_edge) 
        agg_preds_embeddings = self.mlp_predecessors(agg_preds_embeddings)

        operations_idx_by_succs_edge = precedences.edge_index[1]
        succs_by_edge = resources[precedences.edge_index[0]]
        agg_succs_embeddings = torch.zeros((operations.size(0), operations.size(1)), device=operations.device)
        agg_succs_embeddings.index_add_(0, operations_idx_by_succs_edge, succs_by_edge) 
        agg_succs_embeddings = self.mlp_successors(agg_succs_embeddings)

        embedding = torch.zeros((operations.shape[0], self.embedding_size), device=operations.device)
        embedding[1:-1] = self.mlp_combined(torch.cat([agg_preds_embeddings, agg_succs_embeddings, agg_resources_embeddings, agg_materials_embeddings, item_embeddings, self_embeddings], dim=-1))
        return embedding
  
def L1_EMBEDDING_GNN(Module):
    def __init__(self, embedding_size, embedding_hidden_channels, nb_embedding_layers):
        super(L1_EMBEDDING_GNN, self).__init__()
        conf = FeatureConfiguration()
        self.embedding_size = embedding_size
        self.nb_embedding_layers = nb_embedding_layers
        self.material_layers = ModuleList()
        self.resource_layers = ModuleList()
        self.item_layers = ModuleList()
        self.operation_layers = ModuleList()
        self.material_layers.append(MaterialEmbedding(len(conf.material), len(conf.operation), embedding_size))
        self.resource_layers.append(ResourceEmbedding(len(conf.resource), len(conf.operation), embedding_size))
        self.item_layers.append(ItemLayer(len(conf.operation), embedding_size, embedding_hidden_channels, embedding_size))
        self.operation_layers.append(OperationLayer(len(conf.operation), embedding_size, embedding_size, embedding_size, embedding_hidden_channels, embedding_size))
        for _ in range(self.nb_embedding_layers-1):
            self.material_layers.append(MaterialEmbedding(embedding_size, embedding_size, embedding_size))
            self.resource_layers.append(ResourceEmbedding(embedding_size, embedding_size, embedding_size))
            self.item_layers.append(ItemLayer(embedding_size, embedding_size, embedding_hidden_channels, embedding_size))
            self.operation_layers.append(OperationLayer(embedding_size, embedding_size, embedding_size, embedding_size, embedding_hidden_channels, embedding_size))

    def forward(self, state: State, related_items, parents, alpha):
        for l in range(self.nb_embedding_layers):
            state.materials = self.material_layers[l](state.materials, state.operations, state.need_for_materials)
            state.resources = self.resource_layers[l](state.resources, state.operations, state.need_for_resources, state.same_types)
            state.items = self.item_layers[l](state.items, parents, state.operations, state.item_assembly, state.operation_assembly)
            state.operations = self.operation_layers[l](state.operations, related_items, state.materials, state.resources, state.need_for_resources, state.need_for_materials, state.precedences)
        pooled_materials = global_mean_pool(state.materials, torch.zeros(state.materials.shape[0], dtype=torch.long))
        pooled_resources = global_mean_pool(state.resources, torch.zeros(state.resources.shape[0], dtype=torch.long))
        pooled_items = global_mean_pool(state.items, torch.zeros(state.items.shape[0], dtype=torch.long))
        pooled_operations = global_mean_pool(state.operations, torch.zeros(state.operations.shape[0], dtype=torch.long))
        state_embedding = torch.cat([pooled_items, pooled_operations, pooled_materials, pooled_resources, alpha], dim=-1)[0]
        return state, state_embedding

def L1_ACTOR_CRITIC_GNN(Module):
    def __init__(self, shared_embedding_layers: L1_EMBEDDING_GNN, actor_critic_hidden_channels, decision_type):
        super(L1_ACTOR_CRITIC_GNN, self).__init__()
        self.shared_embedding_layers = shared_embedding_layers
        self.decision_type = decision_type
        self.actor_input_size = (self.embedding_size * 5) + 2 if decision_type == OUTSOURCING else (self.embedding_size * 6) + 1
        self.actor = Sequential(
            Linear(self.actor_input_size, actor_critic_hidden_channels), Tanh(),
            Linear(actor_critic_hidden_channels, actor_critic_hidden_channels), Tanh(),
            Linear(actor_critic_hidden_channels, 1)
        )
        self.critic_mlp = Sequential(
            Linear((self.embedding_size * 4) + 1, actor_critic_hidden_channels), Tanh(),
            Linear(actor_critic_hidden_channels, actor_critic_hidden_channels), Tanh(), 
            Linear(actor_critic_hidden_channels, 1)
        )

    def forward(self, state: State, actions, related_items, parents, alpha):
        state, state_embedding = self.shared_embedding_layers(state, related_items, parents, alpha)
        inputs = torch.zeros((len(actions), self.actor_input_size))
        for i, (first_id, second_id) in enumerate(actions.outsourcing):
            if self.decision_type == OUTSOURCING:
                inputs[i] = torch.cat([state.items[first_id], torch.tensor([second_id], dtype=torch.long), state_embedding], dim=-1)
            elif self.decision_type == SCHEDULING:
                inputs[i] = torch.cat([state.operations[first_id], state.resources[second_id], state_embedding], dim=-1)
            else: 
                inputs[i] = torch.cat([state.operations[first_id], state.materials[second_id], state_embedding], dim=-1)
        action_logits = self.actor(inputs)
        action_probs = F.softmax(action_logits, dim=0)
        state_value = self.critic_mlp(state_embedding)
        return action_probs, state_value
