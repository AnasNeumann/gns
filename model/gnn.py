import torch
from torch_geometric.data.storage import EdgeStorage
from torch.nn import Sequential, Linear, ELU, Tanh, Parameter, LeakyReLU, Module, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch import Tensor
from .graph import FeatureConfiguration, State
from torch_geometric.utils import scatter

# =====================================================
# =*= GRAPH ATTENTION NEURAL NETWORK (GaNN) =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

class MaterialEmbeddingLayer(Module):
    def __init__(self, material_dimension: int, operation_dimension: int, embedding_dimension: int):
        super(MaterialEmbeddingLayer, self).__init__()
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

    def forward(self, materials: Tensor, operations: Tensor, need_for_materials: EdgeStorage):
        materials = self.material_transform(materials)
        self_attention = self.leaky_relu(torch.matmul(torch.cat([materials, materials], dim=-1), self.att_self_coef))
        
        ops_by_edges = self.operation_transform(torch.cat([operations[need_for_materials.edge_index[0]], need_for_materials.edge_attr], dim=-1))
        mat_by_edges = materials[need_for_materials.edge_index[1]]
        cross_attention = self.leaky_relu(torch.matmul(torch.cat([mat_by_edges, ops_by_edges], dim=-1), self.att_coef))

        normalizer = F.softmax(torch.cat([self_attention, cross_attention], dim=0), dim=0)
        norm_self_attention = normalizer[:self_attention.size(0)]
        norm_cross_attention = normalizer[self_attention.size(0):]

        weighted_ops_by_edges = norm_cross_attention * ops_by_edges
        sum_ops_by_edges = scatter(weighted_ops_by_edges, need_for_materials.edge_index[1], dim=0, dim_size=materials.size(0))
        
        embedding = F.elu(norm_self_attention * materials + sum_ops_by_edges)
        return embedding

class ResourceEmbeddingLayer(Module):
    def __init__(self, resource_dimension: int, operation_dimension: int, embedding_dimension: int):
        super(ResourceEmbeddingLayer, self).__init__()
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

    def forward(self, resources: Tensor, operations: Tensor, need_for_resources: EdgeStorage, same_types: EdgeStorage):
        self_resources = self.self_transform(resources) 
        self_attention = self.leaky_relu(torch.matmul(torch.cat([self_resources, self_resources], dim=-1), self.att_self_coef))
        ops_by_need_edges = self.operation_transform(torch.cat([operations[need_for_resources.edge_index[0]], need_for_resources.edge_attr], dim=-1))
        res_by_need_edges = self_resources[need_for_resources.edge_index[1]]
        operations_cross_attention = self.leaky_relu(torch.matmul(torch.cat([res_by_need_edges, ops_by_need_edges], dim=-1), self.att_operation_coef))
        
        if same_types:
            res1_by_same_edges = self.resource_transform(resources[same_types.edge_index[0]])
            res2_by_same_edges = self_resources[same_types.edge_index[1]]
            resources_cross_attention = self.leaky_relu(torch.matmul(torch.cat([res2_by_same_edges, res1_by_same_edges], dim=-1), self.att_resource_coef))

            normalizer = F.softmax(torch.cat([self_attention, operations_cross_attention, resources_cross_attention], dim=0), dim=0)
            norm_operations_cross_attention = normalizer[self_attention.size(0):self_attention.size(0)+operations_cross_attention.size(0)]
            norm_resources_cross_attention = normalizer[self_attention.size(0)+operations_cross_attention.size(0):]

            weighted_res_by_edges = norm_resources_cross_attention * res1_by_same_edges
            sum_res_by_edges = scatter(weighted_res_by_edges, same_types.edge_index[1], dim=0, dim_size=self_resources.size(0))

            weighted_ops_by_edges = norm_operations_cross_attention * ops_by_need_edges
            sum_ops_by_edges = scatter(weighted_ops_by_edges, need_for_resources.edge_index[1], dim=0, dim_size=self_resources.size(0))
            embedding = F.elu(normalizer[:self_attention.size(0)] * self_resources + sum_ops_by_edges + sum_res_by_edges)
        else:
            normalizer = F.softmax(torch.cat([self_attention, operations_cross_attention], dim=0), dim=0)
            norm_operations_cross_attention = normalizer[self_attention.size(0):]

            weighted_ops_by_edges = norm_operations_cross_attention * ops_by_need_edges
            sum_ops_by_edges = scatter(weighted_ops_by_edges, need_for_resources.edge_index[1], dim=0, dim_size=self_resources.size(0))
            embedding = F.elu(normalizer[:self_attention.size(0)] * self_resources + sum_ops_by_edges)
        return embedding

class ItemEmbeddingLayer(Module):
    def __init__(self, operation_dimension: int, item_dimension: int, hidden_channels: int, out_channels: int):
        super(ItemEmbeddingLayer, self).__init__()
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

    def forward(self, items: Tensor, parents: Tensor, operations: Tensor, item_assembly: EdgeStorage, operation_assembly: EdgeStorage):
        self_embeddings = self.mlp_self(items)
        parent_embeddings = self.mlp_parent(items[parents])

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
        embedding[:-1] = self.mlp_combined(torch.cat([parent_embeddings[:-1], agg_children_embeddings[:-1], agg_ops_embeddings[:-1], self_embeddings[:-1]], dim=-1))
        return embedding
    
class OperationEmbeddingLayer(Module):
    def __init__(self, operation_dimension: int, item_dimension: int, resources_dimension: int, material_dimension: int, hidden_channels: int, out_channels: int):
        super(OperationEmbeddingLayer, self).__init__()
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

    def forward(self, operations: Tensor, items: Tensor, related_items: Tensor, materials: Tensor, resources: Tensor, need_for_resources: EdgeStorage, need_for_materials: EdgeStorage, precedences: EdgeStorage):
        self_embeddings = self.mlp_self(operations)
        item_embeddings = self.mlp_items(items[related_items])

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
        preds_by_edge = operations[precedences.edge_index[1]]
        agg_preds_embeddings = torch.zeros((operations.size(0), operations.size(1)), device=operations.device)
        agg_preds_embeddings.index_add_(0, operations_idx_by_pred_edge, preds_by_edge) 
        agg_preds_embeddings = self.mlp_predecessors(agg_preds_embeddings)

        operations_idx_by_succs_edge = precedences.edge_index[1]
        succs_by_edge = operations[precedences.edge_index[0]]
        agg_succs_embeddings = torch.zeros((operations.size(0), operations.size(1)), device=operations.device)
        agg_succs_embeddings.index_add_(0, operations_idx_by_succs_edge, succs_by_edge) 
        agg_succs_embeddings = self.mlp_successors(agg_succs_embeddings)

        embedding = torch.zeros((operations.shape[0], self.embedding_size), device=operations.device)
        embedding = self.mlp_combined(torch.cat([agg_preds_embeddings, agg_succs_embeddings, agg_resources_embeddings, agg_materials_embeddings, item_embeddings, self_embeddings], dim=-1))
        return embedding

class L1_EmbbedingGNN(Module):
    def __init__(self, embedding_size: int, embedding_hidden_channels: int, nb_embedding_layers: int):
        super(L1_EmbbedingGNN, self).__init__()
        conf = FeatureConfiguration()
        self.embedding_size = embedding_size
        self.nb_embedding_layers = nb_embedding_layers
        self.material_layers = ModuleList()
        self.resource_layers = ModuleList()
        self.item_layers = ModuleList()
        self.operation_layers = ModuleList()
        self.material_layers.append(MaterialEmbeddingLayer(len(conf.material), len(conf.operation)+len(conf.need_for_materials), embedding_size))
        self.resource_layers.append(ResourceEmbeddingLayer(len(conf.resource), len(conf.operation)+len(conf.need_for_resources), embedding_size))
        self.item_layers.append(ItemEmbeddingLayer(len(conf.operation), len(conf.item), embedding_hidden_channels, embedding_size))
        self.operation_layers.append(OperationEmbeddingLayer(len(conf.operation), embedding_size, embedding_size, embedding_size, embedding_hidden_channels, embedding_size))
        for _ in range(self.nb_embedding_layers-1):
            self.material_layers.append(MaterialEmbeddingLayer(embedding_size, embedding_size+len(conf.need_for_materials), embedding_size))
            self.resource_layers.append(ResourceEmbeddingLayer(embedding_size, embedding_size+len(conf.need_for_resources), embedding_size))
            self.item_layers.append(ItemEmbeddingLayer(embedding_size, embedding_size, embedding_hidden_channels, embedding_size))
            self.operation_layers.append(OperationEmbeddingLayer(embedding_size, embedding_size, embedding_size, embedding_size, embedding_hidden_channels, embedding_size))

    def forward(self, state: State, related_items: Tensor, parents: Tensor, alpha: Tensor):
        for l in range(self.nb_embedding_layers):
            state.materials = self.material_layers[l](state.materials, state.operations, state.need_for_materials)
            state.resources = self.resource_layers[l](state.resources, state.operations, state.need_for_resources, state.same_types)
            state.items = self.item_layers[l](state.items, parents, state.operations, state.item_assembly, state.operation_assembly)
            state.operations = self.operation_layers[l](state.operations, state.items, related_items, state.materials, state.resources, state.need_for_resources, state.need_for_materials, state.precedences)
        pooled_materials = global_mean_pool(state.materials, torch.zeros(state.materials.shape[0], dtype=torch.long, device=state.materials.device))
        pooled_resources = global_mean_pool(state.resources, torch.zeros(state.resources.shape[0], dtype=torch.long, device=state.resources.device))
        pooled_items = global_mean_pool(state.items, torch.zeros(state.items.shape[0], dtype=torch.long, device=state.items.device))
        pooled_operations = global_mean_pool(state.operations, torch.zeros(state.operations.shape[0], dtype=torch.long, device=state.operations.device))
        state_embedding = torch.cat([pooled_items, pooled_operations, pooled_materials, pooled_resources], dim=-1)[0]
        return state, torch.cat([state_embedding, alpha], dim=0)

class L1_CommonCritic(Module):
    def __init__(self, embedding_size: int, critic_hidden_channels: int):
        super(L1_CommonCritic, self).__init__()
        self.critic_mlp = Sequential(
            Linear((embedding_size * 4) + 1, critic_hidden_channels), Tanh(),
            Linear(critic_hidden_channels, critic_hidden_channels), Tanh(), 
            Linear(critic_hidden_channels, 1)
        )

    def forward(self, state_embedding: Tensor):
        return self.critic_mlp(state_embedding)

class L1_OutousrcingActor(Module):
    def __init__(self, shared_embedding_layers: L1_EmbbedingGNN, shared_critic_mlp: L1_CommonCritic, embedding_size: int, actor_hidden_channels: int):
        super(L1_OutousrcingActor, self).__init__()
        self.shared_embedding_layers = shared_embedding_layers
        self.critic_mlp = shared_critic_mlp
        self.actor_input_size = (embedding_size * 5) + 2
        self.actor = Sequential(
            Linear(self.actor_input_size, actor_hidden_channels), Tanh(),
            Linear(actor_hidden_channels, actor_hidden_channels), Tanh(),
            Linear(actor_hidden_channels, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], related_items: Tensor, parents: Tensor, alpha: Tensor):
        state, state_embedding = self.shared_embedding_layers(state, related_items, parents, alpha)
        inputs = torch.zeros((len(actions), self.actor_input_size), device=parents.device)
        for i, (item_id, outsourcing_choice) in enumerate(actions):
            inputs[i] = torch.cat([state.items[item_id], torch.tensor([outsourcing_choice], dtype=torch.long, device=parents.device), state_embedding], dim=-1)
        action_logits = self.actor(inputs)
        action_probs = F.softmax(action_logits, dim=0)
        state_value = self.critic_mlp(state_embedding)
        return action_probs, state_value
    
class L1_SchedulingActor(Module):
    def __init__(self, shared_embedding_layers: L1_EmbbedingGNN,  shared_critic_mlp: L1_CommonCritic, embedding_size: int, actor_hidden_channels: int):
        super(L1_SchedulingActor, self).__init__()
        self.shared_embedding_layers = shared_embedding_layers
        self.critic_mlp = shared_critic_mlp
        self.actor_input_size = (embedding_size * 6) + 1
        self.actor = Sequential(
            Linear(self.actor_input_size, actor_hidden_channels), Tanh(),
            Linear(actor_hidden_channels, actor_hidden_channels), Tanh(),
            Linear(actor_hidden_channels, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], related_items: Tensor, parents: Tensor, alpha: Tensor):
        state, state_embedding = self.shared_embedding_layers(state, related_items, parents, alpha)
        inputs = torch.zeros((len(actions), self.actor_input_size), device=parents.device)
        for i, (operation_id, resource_id) in enumerate(actions):
            inputs[i] = torch.cat([state.operations[operation_id], state.resources[resource_id], state_embedding], dim=-1)
        action_logits = self.actor(inputs)
        action_probs = F.softmax(action_logits, dim=0)
        state_value = self.critic_mlp(state_embedding)
        return action_probs, state_value

class L1_MaterialActor(Module):
    def __init__(self, shared_embedding_layers: L1_EmbbedingGNN, shared_critic_mlp: L1_CommonCritic, embedding_size: int, actor_hidden_channels: int):
        super(L1_MaterialActor, self).__init__()
        self.shared_embedding_layers = shared_embedding_layers
        self.critic_mlp = shared_critic_mlp
        self.actor_input_size = (embedding_size * 6) + 1
        self.actor = Sequential(
            Linear(self.actor_input_size, actor_hidden_channels), Tanh(),
            Linear(actor_hidden_channels, actor_hidden_channels), Tanh(),
            Linear(actor_hidden_channels, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], related_items: Tensor, parents: Tensor, alpha: Tensor):
        state, state_embedding = self.shared_embedding_layers(state, related_items, parents, alpha)
        inputs = torch.zeros((len(actions), self.actor_input_size), device=parents.device)
        for i, (operation_id, material_id) in enumerate(actions):
            inputs[i] = torch.cat([state.operations[operation_id], state.materials[material_id], state_embedding], dim=-1)
        action_logits = self.actor(inputs)
        action_probs = F.softmax(action_logits, dim=0)
        state_value = self.critic_mlp(state_embedding)
        return action_probs, state_value