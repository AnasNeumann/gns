import argparse
import pickle
from torch.multiprocessing import Pool, set_start_method
import os
from model import Instance, GraphInstance, L1_ACTOR_CRITIC_GNN, L1_EMBEDDING_GNN
from common import load_instance, to_bool
import torch
torch.autograd.set_detect_anomaly(True)

PROBLEM_SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
OUTSOURCING = "outsourcing"
SCHEDULING = "scheduling"
MATERIAL_USE = "material_use"
LEARNING_RATE = 2e-4
PARRALLEL = True
DEVICE = None
PPO_CONF = {
    "validation_rate": 10, 
    "switch_batch": 20, 
    "train_iterations": 1000, 
    "opt_epochs": 3,
    "batch_size": 20,
    "clip_ratio": 0.2,
    "policy_loss": 1.0, 
    "value_loss": 0.5, 
    "entropy": 0.01, 
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0,
    'validation_ratio': 0.1
}
GNN_CONF = {
    'embedding_size': 16,
    'nb_layers': 2,
    'hidden_channels': 128
}
AC_CONF = {
    'hidden_channels': 64
}

# =====================================================
# =*= MODEL LOADING METHOD =*=
# =====================================================

def save_models(models):
    for model, name in models:
        torch.save(model.state_dict(), './models/'+name+'_weights.pth')

def load_trained_models():
    shared_GNN = L1_EMBEDDING_GNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load('./models/gnn_weights.pth'))
    outsourcing_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], OUTSOURCING)
    scheduling_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], SCHEDULING)
    material_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], MATERIAL_USE)
    outsourcing_actor.load_state_dict(torch.load('./models/outsourcing_weights.pth'))
    scheduling_actor.load_state_dict(torch.load('./models/scheduling_weights.pth'))
    material_actor.load_state_dict(torch.load('./models/material_weights.pth'))
    return (shared_GNN, 'gnn'), (outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')

def init_new_models():
    shared_GNN = L1_EMBEDDING_GNN(GNN_CONF['embedding_size'], GNN_CONF['hidden_channels'], GNN_CONF['nb_layers'])
    outsourcing_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], OUTSOURCING)
    scheduling_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], SCHEDULING)
    material_actor = L1_ACTOR_CRITIC_GNN(shared_GNN, AC_CONF['hidden_channels'], MATERIAL_USE)
    return (shared_GNN, 'gnn'), (outsourcing_actor, 'outsourcing'), (scheduling_actor, 'scheduling'), (material_actor, 'material')

# =====================================================
# =*= TRANSLATE INSTANCE TO GRAPH =*=
# =====================================================

def translate(i: Instance):
    graph = GraphInstance()
    # TODO
    return graph

# =====================================================
# =*= EXECUTE ONE INSTANCE =*=
# =====================================================

def solve_one(instance: Instance, graph: GraphInstance, shared_GNN: L1_EMBEDDING_GNN, outsourcing_actor: L1_ACTOR_CRITIC_GNN, scheduling_actor: L1_ACTOR_CRITIC_GNN, material_actor: L1_ACTOR_CRITIC_GNN, path, save=False):
    # TODO solve and save results
    pass

# =====================================================
# =*= PPO TRAINING PROCESS =*=
# =====================================================

def load_training_dataset():
    instances = []
    for size in PROBLEM_SIZES:
        problems = []
        path = './instances/train/'+size+'/'
        for i in os.listdir(path):
            if i.endswith('.pkl'):
                file_path = os.path.join(path, i)
                with open(file_path, 'rb') as file:
                    problems.append(pickle.load(file))
        instances.append(problems)
    print("end of loading!")
    return instances

def train(instances, shared_GNN: L1_EMBEDDING_GNN, outsourcing_actor: L1_ACTOR_CRITIC_GNN, scheduling_actor: L1_ACTOR_CRITIC_GNN, material_actor: L1_ACTOR_CRITIC_GNN):
    # TODO train models based on instances then save models
    save_models([shared_GNN, outsourcing_actor, scheduling_actor, material_actor])

# =====================================================
# =*= MAIN CODE =*=
# =====================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII exact solver")
    parser.add_argument("--size", help="Size of the solved instance", required=False)
    parser.add_argument("--id", help="Id of the solved instance", required=False)
    parser.add_argument("--train", help="Do you want to load a pre-trained model", required=True)
    parser.add_argument("--mode", help="Execution mode (either prod or test)", required=True)
    args = parser.parse_args()
    print(f"Execution mode: {args.mode}...")
    if args.mode == 'test':
        PPO_CONF['train_iterations'] = 10
        PPO_CONF['batch_size'] = 3
        PARRALLEL = False
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    if to_bool(args.train):
        print("LOAD DATASET TO TRAIN MODELS")
        instances = load_training_dataset()
        print("TRAIN MODELS WITH PPO")
        shared_GNN, outsourcing_actor, scheduling_actor, material_actor = init_new_models()
        train(instances, shared_GNN, outsourcing_actor, scheduling_actor, material_actor)
    else:
        print("SOLVE A TARGET INSTANCE")
        INSTANCE_PATH = './instances/test/'+args.size+'/instance_'+args.id+'.pkl'
        SOLUTION_PATH = './instances/test/'+args.size+'/solution_'+args.id+'.csv'
        print("Loading "+INSTANCE_PATH+"...")
        instance = load_instance(INSTANCE_PATH)
        shared_GNN, outsourcing_actor, scheduling_actor, material_actor = load_trained_models()
        graph = translate(instance)
        solve_one(instance, graph, shared_GNN, outsourcing_actor, scheduling_actor, material_actor, SOLUTION_PATH, save=True)
    print("===* END OF FILE *===")