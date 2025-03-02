import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.utils import to_dense_adj
from torch import Tensor
from tools.common import num_feature
from tools.tensors import features2tensor, id2tensor
from .instance import Instance

# =============================================================
# =*= HYPER-GRAPH DATA STRUCTURES & MANIPULATION FUNCTIONS =*=
# =============================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

NOT_YET = -1
YES = 1
NO = 0

class State:
    def __init__(self, items: Tensor, operations: Tensor, resources: Tensor, materials: Tensor, need_for_materials: EdgeStorage, need_for_resources: EdgeStorage, operation_assembly: EdgeStorage, item_assembly: EdgeStorage, precedences: EdgeStorage, same_types: EdgeStorage, percentage_outsourced: float=0, percentage_executed_ops: float=0, percentage_executed_items: float=0, nb_projects: int=0, mean_levels: float=0, graph_level_features:Tensor=None, device: str=""):
        self.items: Tensor = items.clone().to(device)
        self.operations: Tensor = operations.clone().to(device)
        self.resources: Tensor = resources.clone().to(device)
        self.materials: Tensor = materials.clone().to(device)
        self.need_for_resources: EdgeStorage = need_for_resources.clone().to(device)
        self.need_for_materials: EdgeStorage = need_for_materials.clone().to(device)
        self.operation_assembly: EdgeStorage = operation_assembly
        self.item_assembly: EdgeStorage = item_assembly
        self.precedences: EdgeStorage = precedences
        self.same_types: EdgeStorage = same_types
        if graph_level_features is None:
            self.graph_level_features: Tensor = State.to_tensor_graph_level_features(len(same_types), len(items), len(operations), len(resources), len(materials), percentage_outsourced, percentage_executed_ops, percentage_executed_items, nb_projects, mean_levels, device)
        else: 
            self.graph_level_features: Tensor = graph_level_features.clone().to(device)

    @staticmethod
    def to_tensor_graph_level_features(nb_same_types: int, nb_items: int, nb_operations: int, nb_resources: int, nb_materials: int, percentage_outsourced: float, percentage_executed_ops: float, percentage_executed_items: float, nb_projects: int, mean_levels: float, device: str):
        return torch.tensor([nb_same_types, nb_items, nb_operations, nb_resources, nb_materials, percentage_outsourced, percentage_executed_ops, percentage_executed_items, nb_projects, mean_levels], dtype=torch.float, device=device)
    
    def clone(self, device: str):
        return State(self.items, self.operations, self.resources, self.materials, self.need_for_materials, self.need_for_resources, self.operation_assembly, self.item_assembly, self.precedences, self.same_types, graph_level_features=self.graph_level_features, device=device)

class FeatureConfiguration:
    def __init__(self):
        self.operation = {
            'design': 0,
            'sync': 1,
            'timescale_hours': 2,
            'timescale_days': 3,
            'direct_successors': 4,
            'total_successors': 5,
            'remaining_time': 6,
            'remaining_resources': 7,
            'remaining_materials': 8,
            'available_time': 9,
            'end_time': 10,
            'is_possible': 11,
            'init_start_time': 12,
            'init_end_time': 13
        }
        self.resource = {
            'utilization_ratio': 0,
            'available_time': 1,
            'executed_operations': 2,
            'remaining_operations': 3,
            'similar_resources': 4
        }
        self.material = {
            'remaining_init_quantity': 0,
            'arrival_time': 1,
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
            'children_time': 10,
            'start_time': 11,
            'end_time': 12,
            'is_possible': 13,
            'init_start_time': 14,
            'init_end_time': 15
        }
        self.need_for_resources = {
            'status': 0,
            'basic_processing_time': 1,
            'current_processing_time': 2,
            'start_time': 3,
            'end_time': 4,
            'init_start_time': 5,
            'init_end_time': 6
        }
        self.need_for_materials = {
            'status': 0,
            'execution_time': 1,
            'quantity_needed': 2,
            'init_execution_time': 3
        }

class OperationFeatures:
    def __init__(self, design: num_feature, sync: num_feature, timescale_hours: num_feature, timescale_days: num_feature, direct_successors: num_feature, total_successors: num_feature, remaining_time: num_feature, remaining_resources: num_feature, remaining_materials: num_feature, available_time: num_feature, end_time: num_feature, is_possible: num_feature, init_start_time: num_feature, init_end_time: num_feature):
        self.design = design
        self.sync = sync
        self.timescale_hours = timescale_hours
        self.timescale_days = timescale_days
        self.direct_successors = direct_successors
        self.total_successors = total_successors
        self.remaining_time = remaining_time
        self.remaining_resources = remaining_resources
        self.remaining_materials = remaining_materials
        self.available_time = available_time
        self.end_time = end_time
        self.is_possible = is_possible
        self.init_start_time = init_start_time
        self.init_end_time = init_end_time
    
    def to_tensor_features(self, device: str):
        return features2tensor([self.design, self.sync, self.timescale_hours, self.timescale_days, self.direct_successors, self.total_successors, self.remaining_time, self.remaining_resources, self.remaining_materials, self.available_time, self.end_time, self.is_possible, self.init_start_time, self.init_end_time], device)
    
class ResourceFeatures:
    def __init__(self, utilization_ratio: num_feature, available_time: num_feature, executed_operations: num_feature, remaining_operations: num_feature, similar_resources: num_feature):
        self.utilization_ratio = utilization_ratio
        self.available_time = available_time
        self.executed_operations = executed_operations
        self.remaining_operations = remaining_operations
        self.similar_resources = similar_resources
    
    def to_tensor_features(self, device: str):
        return features2tensor([self.utilization_ratio, self.available_time, self.executed_operations, self.remaining_operations, self.similar_resources], device)
    
class MaterialFeatures:
    def __init__(self, remaining_init_quantity: num_feature, arrival_time: num_feature, remaining_demand: num_feature):
       self.remaining_init_quantity = remaining_init_quantity
       self.arrival_time = arrival_time
       self.remaining_demand = remaining_demand

    def to_tensor_features(self, device: str):
        return features2tensor([self.remaining_init_quantity, self.arrival_time, self.remaining_demand], device)
    
class ItemFeatures:
    def __init__(self, head: num_feature, external: num_feature, outsourced: num_feature, outsourcing_cost: num_feature, outsourcing_time: num_feature, remaining_physical_time: num_feature, remaining_design_time: num_feature, parents: num_feature, children: num_feature, parents_physical_time: num_feature, children_time: num_feature, start_time: num_feature, end_time: num_feature, is_possible: num_feature, init_start_time: num_feature, init_end_time: num_feature):
        self.head = head
        self.external = external
        self.outsourced = outsourced
        self.outsourcing_cost = outsourcing_cost
        self.outsourcing_time = outsourcing_time
        self.remaining_physical_time = remaining_physical_time
        self.remaining_design_time = remaining_design_time
        self.parents = parents
        self.children = children
        self.parents_physical_time = parents_physical_time
        self.children_time = children_time
        self.start_time = start_time
        self.end_time = end_time
        self.is_possible = is_possible
        self.init_start_time = init_start_time
        self.init_end_time = init_end_time

    def to_tensor_features(self, device: str):
        return features2tensor([self.head, self.external, self.outsourced, self.outsourcing_cost, self.outsourcing_time, self.remaining_physical_time, self.remaining_design_time, self.parents, self.children, self.parents_physical_time, self.children_time, self.start_time, self.end_time, self.is_possible, self.init_start_time, self.init_end_time], device)

    @staticmethod
    def from_tensor(tensor: Tensor, conf: FeatureConfiguration):
        f = conf.item
        return ItemFeatures(
            head=tensor[f['head']].item(), 
            external=tensor[f['external']].item(),
            outsourced=tensor[f['outsourced']].item(),
            outsourcing_cost=tensor[f['outsourcing_cost']].item(),
            outsourcing_time=tensor[f['outsourcing_time']].item(),
            remaining_physical_time=tensor[f['remaining_physical_time']].item(),
            remaining_design_time=tensor[f['remaining_design_time']].item(),
            parents=tensor[f['parents']].item(),
            children=tensor[f['children']].item(),
            start_time=tensor[f['start_time']].item(),
            parents_physical_time=tensor[f['parents_physical_time']].item(),
            children_time=tensor[f['children_time']].item(),
            end_time=tensor[f['end_time']].item(),
            is_possible=tensor[f['is_possible']].item(),
            init_start_time=tensor[f['init_start_time']].item(),
            init_end_time=tensor[f['init_end_time']].item())
    

class NeedForResourceFeatures:
    def __init__(self, status: num_feature, basic_processing_time: num_feature, current_processing_time: num_feature, start_time: num_feature, end_time: num_feature, init_start_time: num_feature, init_end_time: num_feature):
        self.status = status
        self.basic_processing_time = basic_processing_time
        self.current_processing_time = current_processing_time
        self.start_time = start_time
        self.end_time = end_time
        self.init_start_time = init_start_time
        self.init_end_time = init_end_time

    def to_tensor_features(self, device: str):
        return features2tensor([self.status, self.basic_processing_time, self.current_processing_time, self.start_time, self.end_time, self.init_start_time, self.init_end_time], device)

    @staticmethod
    def from_tensor(tensor: Tensor, conf: FeatureConfiguration):
        f = conf.need_for_resources
        return NeedForResourceFeatures(
            status=tensor[f['status']].item(), 
            basic_processing_time=tensor[f['basic_processing_time']].item(),
            current_processing_time=tensor[f['current_processing_time']].item(),
            start_time=tensor[f['start_time']].item(),
            end_time=tensor[f['end_time']].item(),
            init_start_time=tensor[f['init_start_time']].item(),
            init_end_time=tensor[f['init_end_time']].item())

class NeedForMaterialFeatures:
    def __init__(self, status: num_feature, execution_time: num_feature, quantity_needed: num_feature, init_execution_time: num_feature):
        self.status = status
        self.execution_time = execution_time
        self.quantity_needed = quantity_needed
        self.init_execution_time = init_execution_time

    def to_tensor_features(self, device: str):
        return features2tensor([self.status, self.execution_time, self.quantity_needed, self.init_execution_time], device)
    
    @staticmethod
    def from_tensor(tensor: Tensor, conf: FeatureConfiguration):
        f = conf.need_for_materials
        return NeedForMaterialFeatures(
            status=tensor[f['status']].item(), 
            execution_time=tensor[f['execution_time']].item(),
            quantity_needed=tensor[f['quantity_needed']].item(),
            init_execution_time=tensor[f['init_execution_time']].item())

class GraphInstance():
    def __init__(self, device: str):
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
        self.oustourcable_items: int = 0
        self.oustourced_items: int = 0
        self.nb_operations: int = 0
        self.executed_operations: int = 0
        self.executed_items: int = 0
        self.mean_levels: float = 0.0

        self.features = FeatureConfiguration()
        self.graph = HeteroData()
        self.device: str = device
        self.graph.to(device)

    def add_node(self, type: str, features: Tensor):
        self.graph[type].x = torch.cat([self.graph[type].x, features], dim=0) if type in self.graph.node_types else features

    def add_operation(self, p: int, o: int, features: OperationFeatures):
        self.operations_g2i.append((p, o))
        self.add_node('operation', features.to_tensor_features(self.device))
        return len(self.operations_g2i)-1

    def add_item(self, p: int, i: int, features: ItemFeatures):
        self.items_g2i.append((p, i))
        self.add_node('item', features.to_tensor_features(self.device))
        id = len(self.items_g2i)-1
        if features.head == YES:
            self.project_heads.append(id)
        return id
    
    def add_dummy_item(self, device: str):
        self.add_node('item', torch.tensor([[0.0 for _ in range(len(self.features.item))]], dtype=torch.float, device=device))
        dummy_item_id = len(self.graph['item'].x)-1
        for head_id in self.project_heads:
            self.add_item_assembly(dummy_item_id, head_id) 
        for item_id, item in enumerate(self.graph['item'].x):
            if item_id<dummy_item_id and item[self.features.item['children']] == 0:
                self.add_item_assembly(item_id, dummy_item_id)

    def add_material(self, m: int, features: MaterialFeatures):
        self.materials_g2i.append(m)
        self.add_node('material', features.to_tensor_features(self.device))
        return len(self.materials_g2i)-1

    def add_resource(self, r: int, nb_settings: int, features: ResourceFeatures):
        self.resources_g2i.append(r)
        self.current_design_value.append([-1 for _ in range(nb_settings)])
        self.current_operation_type.append(-1)
        self.add_node('resource', features.to_tensor_features(self.device))
        return len(self.resources_g2i)-1

    def add_edge_no_features(self, node_1: str, relation: str, node_2: str, idx: Tensor):
        self.graph[node_1, relation, node_2].edge_index = torch.cat([self.graph[node_1, relation, node_2].edge_index, idx], dim=1) if (node_1, relation, node_2) in self.graph.edge_types else idx

    def add_same_types(self, res_1: int, res_2: int):
        self.add_edge_no_features('resource', 'same', 'resource', id2tensor(res_1, res_2, self.device))
        self.add_edge_no_features('resource', 'same', 'resource', id2tensor(res_2, res_1, self.device))

    def add_item_assembly(self, parent_id: int, child_id: int):
        self.add_edge_no_features('item', 'parent', 'item', id2tensor(parent_id, child_id, self.device))

    def add_operation_assembly(self, item_id: int, operation_id: int):
        self.add_edge_no_features('item', 'has', 'operation', id2tensor(item_id, operation_id, self.device))

    def add_precedence(self, prec_id: int, succ_id: int):
        self.add_edge_no_features('operation', 'precedes', 'operation', id2tensor(prec_id, succ_id, self.device))

    def add_edge_with_features(self, node_1: str, relation: str, node_2: str, idx: Tensor, features: Tensor):
        exists = (node_1, relation, node_2) in self.graph.edge_types
        self.graph[node_1, relation, node_2].edge_index = torch.cat([self.graph[node_1, relation, node_2].edge_index, idx], dim=1) if exists else idx
        self.graph[node_1, relation, node_2].edge_attr = torch.cat([self.graph[node_1, relation, node_2].edge_attr, features], dim=0) if exists else features
    
    def add_need_for_materials(self, operation_id: int, material_id: int, features: NeedForMaterialFeatures):
        self.add_edge_with_features('operation', 'needs_mat', 'material', id2tensor(operation_id, material_id, self.device), features.to_tensor_features(self.device))

    def add_need_for_resources(self, operation_id: int, resource_id: int, features: NeedForResourceFeatures):
        self.add_edge_with_features('operation', 'needs_res', 'resource', id2tensor(operation_id, resource_id, self.device), features.to_tensor_features(self.device))

    def precedences(self):
        return self.graph['operation', 'precedes', 'operation']
    
    def item_assembly(self):
        return self.graph['item', 'parent', 'item']
    
    def operation_assembly(self):
        return self.graph['item', 'has', 'operation']
    
    def need_for_resources(self):
        return self.graph['operation', 'needs_res', 'resource']
    
    def need_for_materials(self):
        return self.graph['operation', 'needs_mat', 'material']
    
    def same_types(self):
        return self.graph['resource', 'same', 'resource']
    
    def operations(self):
        return self.graph['operation'].x
    
    def items(self):
        return self.graph['item'].x
    
    def resources(self):
        return self.graph['resource'].x
    
    def materials(self):
        return self.graph['material'].x

    def operation(self, id: int, feature: str):
        return self.graph['operation'].x[id][self.features.operation[feature]].item()

    def material(self, id: int, feature: str):
        return self.graph['material'].x[id][self.features.material[feature]].item()
    
    def resource(self, id: int, feature: str):
        return self.graph['resource'].x[id][self.features.resource[feature]].item()
    
    def need_for_material(self, operation_id: int, material_id: int, feature: str):
        key = ('operation', 'needs_mat', 'material')
        idx = (self.graph[key].edge_index[0] == operation_id) & (self.graph[key].edge_index[1] == material_id)
        return self.graph[key].edge_attr[idx, self.features.need_for_materials[feature]].item()
    
    def need_for_resource(self, operation_id: int, resource_id: int, feature: str):
        key = ('operation', 'needs_res', 'resource')
        idx = (self.graph[key].edge_index[0] == operation_id) & (self.graph[key].edge_index[1] == resource_id)
        return self.graph[key].edge_attr[idx, self.features.need_for_resources[feature]].item()

    def item(self, id: int, feature: str):
        return self.graph['item'].x[id][self.features.item[feature]].item()
    
    def del_edge(self, edge_type: str, id_1: int, id_2: int):
        edges_idx = self.graph[edge_type].edge_index
        mask = ~((edges_idx[0] == id_1) & (edges_idx[1] == id_2))
        self.graph[edge_type].edge_index = edges_idx[:, mask]
        self.graph[edge_type].edge_attr = self.graph[edge_type].edge_attr[mask]
    
    def del_need_for_resource(self, op_idx: int, res_idx: int):
        self.del_edge(('operation', 'needs_res', 'resource'), op_idx, res_idx)

    def del_need_for_material(self, op_idx: int, mat_idx: int):
        self.del_edge(('operation', 'needs_mat', 'material'), op_idx, mat_idx)

    def update_operation(self, id: int, updates: list[(str, int)], maxx: bool = False):
        for feature, value in updates:
            self.graph['operation'].x[id][self.features.operation[feature]] = value if not maxx else max(value, self.operation(id, feature))
        
    def update_resource(self, id: int, updates: list[(str, int)], maxx: bool = False):
        for feature, value in updates:
            self.graph['resource'].x[id][self.features.resource[feature]] = value if not maxx else max(value, self.resource(id, feature))

    def inc_resource(self, id: int, updates: list[(str, int)]):
        for feature, value in updates:
            self.graph['resource'].x[id][self.features.resource[feature]] += value
    
    def update_material(self, id: int, updates: list[(str, int)], maxx: bool = False):
        for feature, value in updates:
            self.graph['material'].x[id][self.features.material[feature]] = value if not maxx else max(value, self.material(id, feature))

    def inc_material(self, id: int, updates: list[(str, int)]):
        for feature, value in updates:
            self.graph['material'].x[id][self.features.material[feature]] += value
    
    def update_item(self, id: int, updates: list[(str, int)], maxx: bool = False, minn: bool = False):
        for feature, value in updates:
            self.graph['item'].x[id][self.features.item[feature]] = value if not maxx else max(value, self.item(id, feature)) if not minn else min(value, self.item(id, feature))

    def inc_item(self, id: int, updates: list[(str, int)]):
        for feature, value in updates:
            self.graph['item'].x[id][self.features.item[feature]] += value
    
    def inc_operation(self, id: int, updates: list[(str, int)]):
        for feature, value in updates:
            self.graph['operation'].x[id][self.features.operation[feature]] += value

    def update_need_for_material(self, operation_id: int, material_id: int, updates: list[(str, int)], maxx: bool = False):
        key = ('operation', 'needs_mat', 'material')
        idx = (self.graph[key].edge_index[0] == operation_id) & (self.graph[key].edge_index[1] == material_id)
        for feature, value in updates:
            self.graph[key].edge_attr[idx, self.features.need_for_materials[feature]] = value if not maxx else max(value, self.need_for_material(operation_id, material_id, feature))

    def update_need_for_resource(self, operation_id: int, resource_id: int, updates: list[(str, int)], maxx: bool = False):
        key = ('operation', 'needs_res', 'resource')
        idx = (self.graph[key].edge_index[0] == operation_id) & (self.graph[key].edge_index[1] == resource_id)
        for feature, value in updates:
            self.graph[key].edge_attr[idx, self.features.need_for_resources[feature]] = value if not maxx else max(value, self.need_for_resource(operation_id, resource_id, feature))

    def inc_need_for_material(self, operation_id: int, material_id: int, updates: list[(str, int)]):
        key = ('operation', 'needs_mat', 'material')
        idx = (self.graph[key].edge_index[0] == operation_id) & (self.graph[key].edge_index[1] == material_id)
        for feature, value in updates:
            self.graph[key].edge_attr[idx, self.features.need_for_materials[feature]] += value

    def inc_need_for_resource(self, operation_id: int, resource_id: int, updates: list[(str, int)]):
        key = ('operation', 'needs_res', 'resource')
        idx = (self.graph[key].edge_index[0] == operation_id) & (self.graph[key].edge_index[1] == resource_id)
        for feature, value in updates:
            self.graph[key].edge_attr[idx, self.features.need_for_resources[feature]] += value

    def is_item_complete(self, item_id: int):
        if self.item(item_id, 'remaining_physical_time')>0 \
            or self.item(item_id, 'remaining_design_time')>0 \
            or self.item(item_id, 'outsourced')==NOT_YET \
            or self.item(item_id, 'is_possible')==NOT_YET:
            return False
        return True

    def is_operation_complete(self, operation_id: int):
        if self.operation(operation_id, 'is_possible')==NOT_YET \
            or self.operation(operation_id, 'remaining_resources')>0 \
            or self.operation(operation_id, 'remaining_materials')>0 \
            or self.operation(operation_id, 'remaining_time')>0:
            return False
        return True

    def flatten_parents(self, device: str):
        adj = to_dense_adj(self.item_assembly().edge_index)[0]
        nb_items = self.items().size(0)
        parents = torch.zeros(nb_items, dtype=torch.long, device=device)
        for i in range(nb_items-1):
            parents[i] = adj[:,i].nonzero(as_tuple=True)[0]
        return parents

    def flatten_related_items(self, device: str):
        adj = to_dense_adj(self.operation_assembly().edge_index)[0]
        nb_ops = self.operations().size(0)
        r_items = torch.zeros(nb_ops, dtype=torch.long, device=device)
        for i in range(nb_ops):
            r_items[i] = adj[:,i].nonzero(as_tuple=True)[0]
        return r_items

    def build_i2g_2D(self, g2i: list[(int, int)]): # items and operations
        nb_project = max(val[0] for val in g2i) + 1
        i2g = [[] for _ in range(nb_project)]
        for position, (project, op_or_item) in enumerate(g2i):
            while len(i2g[project]) <= op_or_item:
                i2g[project].append(-1)
            i2g[project][op_or_item] = position
        return i2g
    
    def build_i2g_1D(self, g2i: list[int], size: int): # resources and materials
        i2g = [-1] * size
        for id in range(len(g2i)):
            i2g[g2i[id]] = id
        return i2g

    def get_direct_children(self, instance: Instance, item_id):
        p, e = self.items_g2i[item_id]
        children = []
        for child in instance.get_children(p,e,direct=True):
            children.append(self.items_i2g[p][child])
        return children
    
    def loop_resources(self):
        return range(self.graph['resource'].x.size(0))
    
    def loop_items(self):
        return range(self.graph['item'].x.size(0)-1)
    
    def loop_operations(self):
        return range(self.graph['operation'].x.size(0))
    
    def loop_materials(self):
        return range(self.graph['material'].x.size(0))
    
    def loop_need_for_material(self):
        key = ('operation', 'needs_mat', 'material')
        return self.graph[key].edge_index, self.graph[key].edge_attr, range(self.graph['operation', 'needs_mat', 'material'].edge_index.size(1))
    
    def loop_need_for_resource(self):
        key = ('operation', 'needs_res', 'resource')
        return self.graph[key].edge_index, self.graph[key].edge_attr, range(self.graph['operation', 'needs_res', 'resource'].edge_index.size(1))
    
    def to_state(self, device: str) -> State:
        return State(items = self.items(), 
                     operations = self.operations(), 
                     resources = self.resources(), 
                     materials = self.materials(), 
                     need_for_materials = self.need_for_materials(),
                     need_for_resources = self.need_for_resources(),
                     operation_assembly = self.operation_assembly(),
                     item_assembly = self.item_assembly(),
                     precedences = self.precedences(),
                     same_types = self.same_types(),
                     percentage_outsourced= self.oustourced_items / self.oustourcable_items,
                     percentage_executed_ops = self.executed_operations / self.nb_operations,
                     percentage_executed_items = self.executed_items / len(self.items()),
                     nb_projects = len(self.project_heads),
                     mean_levels = self.mean_levels,
                     device = device)