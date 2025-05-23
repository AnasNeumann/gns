import torch
from torch_geometric.data.storage import EdgeStorage
from torch.nn import Sequential, Linear, ReLU, Parameter, LeakyReLU, Module, ModuleList
import torch.nn.functional as F
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
        self.material_upscale = Linear(material_dimension, embedding_dimension, bias=False)
        self.att_self_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.operation_upscale = Linear(operation_dimension, embedding_dimension, bias=False)
        self.att_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.leaky_relu = LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.material_upscale.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.operation_upscale.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_coef.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_self_coef.data, gain=1.414)

    def forward(self, materials: Tensor, operations: Tensor, need_for_materials: EdgeStorage):
        upscaled_materials = self.material_upscale(materials)
        self_attention = self.leaky_relu(torch.matmul(torch.cat([upscaled_materials, upscaled_materials], dim=-1), self.att_self_coef))
        
        ops_by_edges = self.operation_upscale(torch.cat([operations[need_for_materials.edge_index[0]], need_for_materials.edge_attr], dim=-1))
        mat_by_edges = upscaled_materials[need_for_materials.edge_index[1]]
        cross_attention = self.leaky_relu(torch.matmul(torch.cat([mat_by_edges, ops_by_edges], dim=-1), self.att_coef))

        normalizer = F.softmax(torch.cat([self_attention, cross_attention], dim=0), dim=0)
        norm_self_attention = normalizer[:self_attention.size(0)]
        norm_cross_attention = normalizer[self_attention.size(0):]

        weighted_ops_by_edges = norm_cross_attention * ops_by_edges
        sum_ops_by_edges = scatter(weighted_ops_by_edges, need_for_materials.edge_index[1], dim=0, dim_size=materials.size(0))
        
        embedding = F.elu(norm_self_attention * upscaled_materials + sum_ops_by_edges)
        return embedding

class ResourceEmbeddingLayer(Module):
    def __init__(self, resource_dimension: int, operation_dimension: int, embedding_dimension: int):
        super(ResourceEmbeddingLayer, self).__init__()
        self.self_upscale = Linear(resource_dimension, embedding_dimension, bias=False)
        self.resource_upscale = Linear(resource_dimension, embedding_dimension, bias=False)
        self.operation_upscale = Linear(operation_dimension, embedding_dimension, bias=False)
        self.att_operation_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.att_resource_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.att_self_coef = Parameter(torch.zeros(size=(2 * embedding_dimension, 1)))
        self.leaky_relu = LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.self_upscale.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.resource_upscale.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.operation_upscale.weight.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_operation_coef.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_resource_coef.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_self_coef.data, gain=1.414)

    def forward(self, resources: Tensor, operations: Tensor, need_for_resources: EdgeStorage, same_types: EdgeStorage):
        self_resources_upscaled = self.self_upscale(resources) 
        self_attention = self.leaky_relu(torch.matmul(torch.cat([self_resources_upscaled, self_resources_upscaled], dim=-1), self.att_self_coef))
        ops_by_need_edges = self.operation_upscale(torch.cat([operations[need_for_resources.edge_index[0]], need_for_resources.edge_attr], dim=-1))
        res_by_need_edges = self_resources_upscaled[need_for_resources.edge_index[1]]
        operations_cross_attention = self.leaky_relu(torch.matmul(torch.cat([res_by_need_edges, ops_by_need_edges], dim=-1), self.att_operation_coef))
        
        normalizer = F.softmax(torch.cat([self_attention, operations_cross_attention], dim=0), dim=0)
        norm_operations_cross_attention = normalizer[self_attention.size(0):]

        weighted_ops_by_edges = norm_operations_cross_attention * ops_by_need_edges
        sum_ops_by_edges = scatter(weighted_ops_by_edges, need_for_resources.edge_index[1], dim=0, dim_size=self_resources_upscaled.size(0))
        embedding = F.elu(normalizer[:self_attention.size(0)] * self_resources_upscaled + sum_ops_by_edges)
        return embedding

class ItemEmbeddingLayer(Module):
    def __init__(self, operation_dimension: int, item_dimension: int, hidden_channels: int, out_channels: int):
        super(ItemEmbeddingLayer, self).__init__()
        self.embedding_size = out_channels
        first_dimension = hidden_channels
        second_dimension = int(hidden_channels/2)
        self.mlp_combined = Sequential(
            Linear(4 * out_channels, first_dimension), ReLU(),
            Linear(first_dimension, second_dimension), ReLU(),
            Linear(second_dimension, out_channels)
        )
        self.mlp_operations = Sequential(
            Linear(operation_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )
        self.mlp_parent = Sequential(
            Linear(item_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )
        self.mlp_children = Sequential(
            Linear(item_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )
        self.mlp_self = Sequential(
            Linear(item_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )

    def forward(self, items: Tensor, parents: Tensor, operations: Tensor, item_assembly: EdgeStorage, operation_assembly: EdgeStorage):
        self_embeddings = self.mlp_self(items)
        parent_embeddings = self.mlp_parent(items[parents])

        parent_idx_by_edge = item_assembly.edge_index[0]
        children_by_edge = items[item_assembly.edge_index[1]]
        agg_children = scatter(children_by_edge, parent_idx_by_edge, dim=0, dim_size=items.size(0))
        agg_children_embeddings = self.mlp_children(agg_children)

        item_idx_by_edges = operation_assembly.edge_index[0]
        operations_by_edges = operations[operation_assembly.edge_index[1]]
        agg_ops = scatter(operations_by_edges, item_idx_by_edges, dim=0, dim_size=items.size(0))
        agg_ops_embeddings = self.mlp_operations(agg_ops)
        
        embedding = torch.zeros((items.shape[0], self.embedding_size), device=items.device)
        embedding[:-1] = self.mlp_combined(torch.cat([parent_embeddings[:-1], agg_children_embeddings[:-1], agg_ops_embeddings[:-1], self_embeddings[:-1]], dim=-1))
        return embedding
    
class OperationEmbeddingLayer(Module):
    def __init__(self, operation_dimension: int, item_dimension: int, resources_dimension: int, material_dimension: int, hidden_channels: int, out_channels: int):
        super(OperationEmbeddingLayer, self).__init__()
        self.embedding_size = out_channels
        first_dimension = hidden_channels
        second_dimension = int(hidden_channels/2)
        self.mlp_combined = Sequential(
            Linear(4 * out_channels + resources_dimension + material_dimension, first_dimension), ReLU(),
            Linear(first_dimension, second_dimension), ReLU(),
            Linear(second_dimension, out_channels)
        )
        self.mlp_items = Sequential(
            Linear(item_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )
        self.mlp_predecessors = Sequential(
            Linear(operation_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )
        self.mlp_successors = Sequential(
            Linear(operation_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )
        self.mlp_resources = Sequential(
            Linear(resources_dimension, first_dimension), ReLU(),
            Linear(first_dimension, resources_dimension)
        )
        self.mlp_materials = Sequential(
            Linear(material_dimension, first_dimension), ReLU(),
            Linear(first_dimension, material_dimension)
        )
        self.mlp_self = Sequential(
            Linear(operation_dimension, first_dimension), ReLU(),
            Linear(first_dimension, out_channels)
        )

    def forward(self, operations: Tensor, items: Tensor, related_items: Tensor, materials: Tensor, resources: Tensor, need_for_resources: EdgeStorage, need_for_materials: EdgeStorage, precedences: EdgeStorage):
        self_embeddings = self.mlp_self(operations)
        item_embeddings = self.mlp_items(items[related_items])

        operations_idx_by_mat_edge = need_for_materials.edge_index[0]
        materials_by_edge = materials[need_for_materials.edge_index[1]]
        agg_materials = scatter(materials_by_edge, operations_idx_by_mat_edge, dim=0, dim_size=operations.size(0))
        agg_materials_embeddings = self.mlp_materials(agg_materials)

        operations_idx_by_res_edge = need_for_resources.edge_index[0]
        resources_by_edge = resources[need_for_resources.edge_index[1]]
        agg_resources = scatter(resources_by_edge, operations_idx_by_res_edge, dim=0, dim_size=operations.size(0))
        agg_resources_embeddings = self.mlp_resources(agg_resources)

        operations_idx_by_pred_edge = precedences.edge_index[0]
        preds_by_edge = operations[precedences.edge_index[1]]
        agg_preds = scatter(preds_by_edge, operations_idx_by_pred_edge, dim=0, dim_size=operations.size(0))
        agg_preds_embeddings = self.mlp_predecessors(agg_preds)

        operations_idx_by_succs_edge = precedences.edge_index[1]
        succs_by_edge = operations[precedences.edge_index[0]]
        agg_succs = scatter(succs_by_edge, operations_idx_by_succs_edge, dim=0, dim_size=operations.size(0))
        agg_succs_embeddings = self.mlp_successors(agg_succs)

        embedding = self.mlp_combined(torch.cat([agg_preds_embeddings, agg_succs_embeddings, agg_resources_embeddings, agg_materials_embeddings, item_embeddings, self_embeddings], dim=-1))
        return embedding

class L1_EmbbedingGNN(Module):
    def __init__(self, resource_and_material_embedding_size: int, operation_and_item_embedding_size: int, embedding_hidden_channels: int, nb_embedding_layers: int):
        super(L1_EmbbedingGNN, self).__init__()
        conf = FeatureConfiguration()
        self.resource_and_material_embedding_size = resource_and_material_embedding_size
        self.operation_and_item_embedding_size = operation_and_item_embedding_size
        self.nb_embedding_layers = nb_embedding_layers
        self.material_layers = ModuleList()
        self.resource_layers = ModuleList()
        self.item_layers = ModuleList()
        self.operation_layers = ModuleList()
        self.material_layers.append(MaterialEmbeddingLayer(len(conf.material), len(conf.operation)+len(conf.need_for_materials), resource_and_material_embedding_size))
        self.resource_layers.append(ResourceEmbeddingLayer(len(conf.resource), len(conf.operation)+len(conf.need_for_resources), resource_and_material_embedding_size))
        self.item_layers.append(ItemEmbeddingLayer(len(conf.operation), len(conf.item), embedding_hidden_channels, operation_and_item_embedding_size))
        self.operation_layers.append(OperationEmbeddingLayer(len(conf.operation), operation_and_item_embedding_size, resource_and_material_embedding_size, resource_and_material_embedding_size, embedding_hidden_channels, operation_and_item_embedding_size))
        for _ in range(self.nb_embedding_layers-1):
            self.material_layers.append(MaterialEmbeddingLayer(resource_and_material_embedding_size, operation_and_item_embedding_size+len(conf.need_for_materials), resource_and_material_embedding_size))
            self.resource_layers.append(ResourceEmbeddingLayer(resource_and_material_embedding_size, operation_and_item_embedding_size+len(conf.need_for_resources), resource_and_material_embedding_size))
            self.item_layers.append(ItemEmbeddingLayer(operation_and_item_embedding_size, operation_and_item_embedding_size, embedding_hidden_channels, operation_and_item_embedding_size))
            self.operation_layers.append(OperationEmbeddingLayer(operation_and_item_embedding_size, operation_and_item_embedding_size, resource_and_material_embedding_size, resource_and_material_embedding_size, embedding_hidden_channels, operation_and_item_embedding_size))

    def forward(self, state: State, related_items: Tensor, parents: Tensor, alpha: Tensor):
        m_embeddings = self.material_layers[0](state.materials, state.operations, state.need_for_materials)
        r_embbedings = self.resource_layers[0](state.resources, state.operations, state.need_for_resources, state.same_types)
        i_embbedings = self.item_layers[0](state.items, parents, state.operations, state.item_assembly, state.operation_assembly)
        o_embbedings = self.operation_layers[0](state.operations, i_embbedings, related_items, m_embeddings, r_embbedings, state.need_for_resources, state.need_for_materials, state.precedences) 
        for l in range(1, self.nb_embedding_layers):
            m_embeddings = self.material_layers[l](m_embeddings, o_embbedings, state.need_for_materials)
            r_embbedings = self.resource_layers[l](r_embbedings, o_embbedings, state.need_for_resources, state.same_types)
            i_embbedings = self.item_layers[l](i_embbedings, parents, o_embbedings, state.item_assembly, state.operation_assembly)
            o_embbedings = self.operation_layers[l](o_embbedings, i_embbedings, related_items, m_embeddings, r_embbedings, state.need_for_resources, state.need_for_materials, state.precedences)     
        pooled_resources = torch.mean(r_embbedings, dim=0, keepdim=True)
        pooled_items = torch.mean(i_embbedings, dim=0, keepdim=True)
        pooled_operations = torch.mean(o_embbedings, dim=0, keepdim=True)
        state_embedding = torch.cat([pooled_items, pooled_operations, pooled_resources], dim=-1)[0]
        return torch.cat([state_embedding, alpha], dim=0), m_embeddings, r_embbedings, i_embbedings, o_embbedings

class L1_CommonCritic(Module):
    def __init__(self, resource_and_material_embedding_size: int, operation_and_item_embedding_size: int, critic_hidden_channels: int):
        super(L1_CommonCritic, self).__init__()
        first_dimension = critic_hidden_channels
        second_dimenstion = int(critic_hidden_channels / 2)
        state_vector_size = resource_and_material_embedding_size + 2*operation_and_item_embedding_size + 1
        self.critic_mlp = Sequential(
            Linear(state_vector_size, first_dimension), ReLU(),
            Linear(first_dimension, second_dimenstion), ReLU(), 
            Linear(second_dimenstion, 1)
        )

    def forward(self, state_embedding: Tensor):
        return self.critic_mlp(state_embedding)

class L1_OutousrcingActor(Module):
    def __init__(self, shared_embedding_layers: L1_EmbbedingGNN, shared_critic_mlp: L1_CommonCritic, resource_and_material_embedding_size: int, operation_and_item_embedding_size: int, actor_hidden_channels: int):
        super(L1_OutousrcingActor, self).__init__()
        self.shared_embedding_layers = shared_embedding_layers
        self.critic_mlp = shared_critic_mlp
        self.actor_input_size = resource_and_material_embedding_size + 3*operation_and_item_embedding_size + 2
        first_dimension = actor_hidden_channels
        second_dimenstion = int(actor_hidden_channels / 2)
        self.actor = Sequential(
            Linear(self.actor_input_size, first_dimension), ReLU(),
            Linear(first_dimension, second_dimenstion), ReLU(),
            Linear(second_dimenstion, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], related_items: Tensor, parents: Tensor, alpha: Tensor):
        state_embedding, _, _, i_embbedings, _ = self.shared_embedding_layers(state, related_items, parents, alpha)
        item_ids, outsourcing_choices = zip(*actions)
        outsourcing_choices_tensor = torch.tensor(outsourcing_choices, dtype=torch.float32, device=parents.device).unsqueeze(1)
        state_embedding_expanded = state_embedding.unsqueeze(0).expand(len(actions), -1)
        inputs = torch.cat([i_embbedings[list(item_ids)], outsourcing_choices_tensor, state_embedding_expanded], dim=1)
        action_logits = self.actor(inputs)
        action_probs = F.softmax(action_logits, dim=0)
        state_value = self.critic_mlp(state_embedding)
        return action_probs, state_value
    
class L1_SchedulingActor(Module):
    def __init__(self, shared_embedding_layers: L1_EmbbedingGNN,  shared_critic_mlp: L1_CommonCritic, resource_and_material_embedding_size: int, operation_and_item_embedding_size: int, actor_hidden_channels: int):
        super(L1_SchedulingActor, self).__init__()
        self.shared_embedding_layers = shared_embedding_layers
        self.critic_mlp = shared_critic_mlp
        self.actor_input_size = 2*resource_and_material_embedding_size + 3*operation_and_item_embedding_size + 1
        first_dimension = actor_hidden_channels
        second_dimenstion = int(actor_hidden_channels / 2)
        self.actor = Sequential(
            Linear(self.actor_input_size, first_dimension), ReLU(),
            Linear(first_dimension, second_dimenstion), ReLU(),
            Linear(second_dimenstion, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], related_items: Tensor, parents: Tensor, alpha: Tensor):
        state_embedding, _, r_embbedings, _, o_embbedings = self.shared_embedding_layers(state, related_items, parents, alpha)
        operations_ids, resources_ids = zip(*actions)
        state_embedding_expanded = state_embedding.unsqueeze(0).expand(len(actions), -1)
        inputs = torch.cat([o_embbedings[list(operations_ids)], r_embbedings[list(resources_ids)], state_embedding_expanded], dim=1) # shape = [possible decision, concat embedding]
        action_logits = self.actor(inputs)
        action_probs = F.softmax(action_logits, dim=0)
        state_value = self.critic_mlp(state_embedding)
        return action_probs, state_value

class L1_MaterialActor(Module):
    def __init__(self, shared_embedding_layers: L1_EmbbedingGNN, shared_critic_mlp: L1_CommonCritic, resource_and_material_embedding_size: int, operation_and_item_embedding_size: int, actor_hidden_channels: int):
        super(L1_MaterialActor, self).__init__()
        self.shared_embedding_layers = shared_embedding_layers
        self.critic_mlp = shared_critic_mlp
        self.actor_input_size = 2*resource_and_material_embedding_size + 3*operation_and_item_embedding_size + 1
        first_dimension = actor_hidden_channels
        second_dimenstion = int(actor_hidden_channels / 2)
        self.actor = Sequential(
            Linear(self.actor_input_size, first_dimension), ReLU(),
            Linear(first_dimension, second_dimenstion), ReLU(),
            Linear(second_dimenstion, 1)
        )

    def forward(self, state: State, actions: list[(int, int)], related_items: Tensor, parents: Tensor, alpha: Tensor):
        state_embedding, m_embeddings, _, _, o_embbedings = self.shared_embedding_layers(state, related_items, parents, alpha)
        operations_ids, materials_ids = zip(*actions)
        state_embedding_expanded = state_embedding.unsqueeze(0).expand(len(actions), -1)
        inputs = torch.cat([o_embbedings[list(operations_ids)], m_embeddings[list(materials_ids)], state_embedding_expanded], dim=1)
        action_logits = self.actor(inputs)
        action_probs = F.softmax(action_logits, dim=0)
        state_value = self.critic_mlp(state_embedding)
        return action_probs, state_value