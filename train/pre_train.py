from model.graph import State
import torch.nn.functional as F
import torch
import os
import pickle
from model.graph import FullState, FeatureConfiguration
import argparse
from model.gnn import L1_EmbbedingGNN, L1_AutoEncoder
import random
from tools.common import directory

# ===========================================================
# =*= PRE-TRAINING USING SUPERVISED AUTO-ENCODER MODEL =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

# 0. Configuration
LEARNING_RATE = 5e-3
EPOCHS = 220
BATCH_SIZE = 64
GNN_CONF = {
    'resource_and_material_embedding_size': 16,
    'operation_and_item_embedding_size': 24,
    'nb_layers': 2,
    'embedding_hidden_channels': 128,
    'decoder_hidden_layers': 128,
}

# I. Custom MSE-based loss function
def reconstruction_loss_with_context(reconstructed: State, original: State):
    material_loss = F.mse_loss(reconstructed.materials, original.materials)
    resource_loss = F.mse_loss(reconstructed.resources, original.resources)
    item_loss = F.mse_loss(reconstructed.items, original.items)
    operation_loss = F.mse_loss(reconstructed.operations, original.operations)
    return material_loss + resource_loss + item_loss + operation_loss

# II. Load randomized dataset
def load_dataset(folder_path) -> list[FullState]:
    print("1 - Loading dataset.....")
    dataset: list[FullState] = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as file:
                obj: FullState = pickle.load(file)
                dataset.append(obj)
    random.shuffle(dataset)
    print("Dataset loaded!")
    return dataset

# III. Build two models
def init_new_models(device: str):
    print("2 - Building models.....")
    conf = FeatureConfiguration()
    encoder: L1_EmbbedingGNN = L1_EmbbedingGNN(GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['embedding_hidden_channels'], GNN_CONF['nb_layers'])
    decoder: L1_AutoEncoder = L1_AutoEncoder(encoder, len(conf.material), len(conf.resource), len(conf.item), len(conf.operation), GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['operation_and_item_embedding_size'], GNN_CONF['resource_and_material_embedding_size'], GNN_CONF['decoder_hidden_layers'])
    encoder.to(device)
    decoder.to(device)
    print("Models ready!")
    return encoder, decoder

# IV. Supervised training function
def train(encoder: L1_EmbbedingGNN, decoder: L1_AutoEncoder, dataset: list[FullState], device: str, path: str):
    print("3 - Start training.....")
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    def data_loader(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

    decoder.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        random.shuffle(dataset)
        for batch in data_loader(dataset, BATCH_SIZE):
            reconstructed_states = []
            for partial_solution in batch:
                reconstructed_state = decoder(partial_solution.state.clone(device), partial_solution.related_items, partial_solution.parents, partial_solution.alpha)
                reconstructed_states.append(reconstructed_state)
            batch_loss = 0.0
            for reconstructed_state, original_state in zip(reconstructed_states, batch):
                batch_loss += reconstruction_loss_with_context(reconstructed_state, original_state)
            batch_loss /= BATCH_SIZE
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            if epoch % 20 == 0:
                save_models(encoder, decoder, path)
        print(f"\t Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.6f}")
    print("End of training!")
    return encoder, decoder

# V. Save models
def save_models(encoder: L1_EmbbedingGNN, decoder: L1_AutoEncoder, complete_path: str):
    torch.save(encoder.state_dict(), complete_path+'/gnn_weights_0.pth')
    torch.save(decoder.state_dict(), complete_path+'/decoder_weights_0.pth')

# VI. Main code
'''
    Test inference mode with: bash _env.sh
    python pre_train.py --path=./
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII/L1 GNS pretraining")
    parser.add_argument("--path", help="Saving path on the server", required=True)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TPU Device selected: {device}...")
    dataset = load_dataset(args.path+directory.states)
    encoder, decoder = init_new_models(device)
    encoder, decoder = train(encoder, decoder, dataset, device, args.path+directory.models)
    save_models(encoder, decoder, args.path+directory.models)
    print("===* END OF FILE *===")

    