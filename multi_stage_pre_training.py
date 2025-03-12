import pickle
import os
from model.instance import Instance
from tools.common import directory, freeze, unfreeze, unfreeze_all, freeze_several
import torch
torch.autograd.set_detect_anomaly(True)
import random
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from debug.debug_gns import debug_printer
from typing import Callable
from model.agent import MultiAgent_OneInstance, MultiAgents_Batch, MAPPO_Loss, MAPPO_Losses
import time as systime
from debug.loss_tracker import LossTracker

# ===========================================================
# =*= PROXIMAL POLICY OPTIMIZATION (PPO) RELATE FUNCTIONS =*=
# ===========================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

PROBLEM_SIZES = [['s'], ['s', 'm']] # ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
PPO_CONF = {
    "validation_rate": 20,
    "switch_batch": 10,
    "train_iterations": [300, 150, 150, 150, 300],
    "opt_epochs": 3,
    "batch_size": 20,
    "clip_ratio": 0.15,
    "policy_loss": 1.0,
    "value_loss": 0.5,
    "entropy": 0.02,
    "discount_factor": 1.0,
    "bias_variance_tradeoff": 1.0
}
LEARNING_RATE = 1e-4
AGENTS = ["outsourcing", "scheduling", "material_use"]
AGENT = 0
OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2

def save_models(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, optimizer: Optimizer, run_number:int, complete_path: str):
    index = str(run_number)
    torch.save(embedding_stack.state_dict(), complete_path+'/gnn_weights_'+index+'.pth')
    torch.save(shared_critic.state_dict(), complete_path+'/critic_weights_'+index+'.pth')
    torch.save(optimizer.state_dict(), complete_path+'/adam_'+index+'.pth')
    for agent, name in agents:
        torch.save(agent.state_dict(), complete_path+'/'+name+'_weights_'+index+'.pth')

def search_instance(instances: list[Instance], id: int) -> Instance:
    for instance in instances:
        if instance.id == id:
            return instance
    return None

def load_training_dataset(debug_mode: bool, path: str, train: bool = True):
    type: str = '/train/' if train else '/test/'
    instances = []
    for size in PROBLEM_SIZES[0 if debug_mode else 1]:
        complete_path = path+directory.instances+type+size+'/'
        for i in os.listdir(complete_path):
            if i.endswith('.pkl'):
                file_path = os.path.join(complete_path, i)
                with open(file_path, 'rb') as file:
                    instances.append(pickle.load(file))
    print(f"End of loading {len(instances)} instances!")
    return instances

def train_or_validate_batch(agents: list[(Module, str)], batch: list[Instance], agent_names: list[str], train: bool, epochs: int, optimizer: Optimizer, solve_function: Callable, device: str, debug: bool, trainable: list, fixed_alpha: float = -1):
    """
        Train or validate on a batch of instances
    """
    replay_memory: list[MultiAgent_OneInstance] = []
    for instance in batch:
        print(f"\t start solving instance: {instance.id}...")
        transitions,_,_,_ = solve_function(instance=instance, agents=agents, train=True, trainable=[True for _ in agents], device=device, debug_mode=debug, trainable=trainable, fixed_alpha=fixed_alpha)
        replay_memory.append(transitions)
    batch_result: MultiAgents_Batch = MultiAgents_Batch(
        batch=replay_memory, 
        agent_names=agent_names, 
        gamma=PPO_CONF['discount_factor'], 
        lam=PPO_CONF['bias_variance_tradeoff'],
        weight_policy_loss=PPO_CONF['policy_loss'],
        weight_value_loss=PPO_CONF['value_loss'], 
        weight_entropy_bonus=PPO_CONF['entropy'],
        clipping_ratio=PPO_CONF['clip_ratio'])
    if train:
        for e in range(epochs):
            print(f"\t Optimization epoch: {e+1}/{epochs}")
            optimizer.zero_grad()
            training_loss: Tensor = batch_result.compute_losses(agents, return_details=False)
            print(f"\t Multi-agent batch loss: {training_loss} - Differentiable computation graph = {training_loss.requires_grad}!")
            training_loss.backward(retain_graph=False)
            optimizer.step()
    else:
        current_vloss, current_details = batch_result.compute_losses(agents, return_details=True)
        print(f'\t Multi-agent batch loss: {current_vloss:.4f}')
        return current_details

def scheduling_stage(train_data: list[Instance], val_data: list[Instance], agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, switch_batch: int, validation_rate: int, device: str, interactive: bool=False, debug_mode: bool=False):
    """
        First (1st) optimization stage: scheduling agent alone
    """
    scheduling_optimizer = torch.optim.Adam(list(agents[SCHEDULING][AGENT].parameters()), lr=LEARNING_RATE)
    debug_print: Callable = debug_printer(debug_mode)
    vlosses = MAPPO_Losses(agent_names=[AGENTS[SCHEDULING]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _scheduling_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Scheduling loss (policy)", title="Scheduling loss (policy)", color="green", show=interactive)
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % switch_batch == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch: list[Instance] = random.sample(train_data, batch_size)
        train_or_validate_batch(agents, current_batch, train=True, epochs=epochs, agent_names=[name for _,name in agents], optimizer=scheduling_optimizer, agent_names=[AGENTS[SCHEDULING]], solve_function=solve_function, device=device, debug=debug_mode, trainable=[0,1,0], fixed_alpha=1.0)
        if iteration % validation_rate == 0:
            debug_print("\t time to validate the loss...")
            agents[SCHEDULING][AGENT].eval()
            with torch.no_grad():
                current_vloss: MAPPO_Loss = train_or_validate_batch(agents, val_data, train=False, epochs=-1, optimizer=None, solve_function=solve_function, agent_names=[AGENTS[SCHEDULING]], device=device, debug=debug_mode, trainable=[0,1,0], fixed_alpha=1.0)
                vlosses.add(current_vloss)
                _vloss_TRACKER.update(current_vloss.value_loss)
                _scheduling_loss_TRACKER.update(current_vloss.get(AGENTS[SCHEDULING]).policy_loss)
            agents[SCHEDULING][AGENT].train()
    complete_path = path + directory.models
    _vloss_TRACKER.save(complete_path+'/stage_'+str(1)+'_value_loss')
    _scheduling_loss_TRACKER.save(complete_path+'/stage_'+str(1)+'_scheduling_loss')
    with open(complete_path+'/validation_'+str(1)+'.pkl', 'wb') as f:
        pickle.dump(vlosses, f)
    save_models(agents=agents, embedding_stack=embedding_stack, shared_critic=shared_critic, optimizer=scheduling_optimizer, run_number=1, complete_path=complete_path)

def material_use_stage(train_data: list[Instance], val_data: list[Instance], agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, switch_batch: int, validation_rate: int, device: str, interactive: bool=False, debug_mode: bool=False):
    """
        Second (2nd) optimization stage: material use agent alone
    """
    material_optimizer = torch.optim.Adam(list(agents[MATERIAL_USE][AGENT].parameters()),  lr=LEARNING_RATE)
    debug_print: Callable = debug_printer(debug_mode)
    vlosses = MAPPO_Losses(agent_names=[AGENTS[MATERIAL_USE]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _material_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Material use loss (policy)", title="Material use loss (policy)", color="yellow", show=interactive)
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % switch_batch == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch: list[Instance] = random.sample(train_data, batch_size)
        train_or_validate_batch(agents, current_batch, train=True, epochs=epochs, optimizer=material_optimizer, solve_function=solve_function, agent_names=[AGENTS[MATERIAL_USE]], device=device, debug=debug_mode, trainable=[0,0,1], fixed_alpha=1.0)
        if iteration % validation_rate == 0:
            debug_print("\t time to validate the loss...")
            agents[MATERIAL_USE][AGENT].eval()
            with torch.no_grad():
                current_vloss: MAPPO_Loss = train_or_validate_batch(agents, val_data, train=False, epochs=-1, optimizer=None, agent_names=[AGENTS[MATERIAL_USE]], solve_function=solve_function, device=device, debug=debug_mode, trainable=[0,0,1], fixed_alpha=1.0)
                vlosses.add(current_vloss)
                _vloss_TRACKER.update(current_vloss.value_loss)
                _material_loss_TRACKER.update(current_vloss.get(AGENTS[MATERIAL_USE]).policy_loss)
            agents[MATERIAL_USE][AGENT].train()
    complete_path = path + directory.models
    _vloss_TRACKER.save(complete_path+'/stage_'+str(2)+'_value_loss')
    _material_loss_TRACKER.save(complete_path+'/stage_'+str(2)+'_material_loss')
    with open(complete_path+'/validation_'+str(2)+'.pkl', 'wb') as f:
        pickle.dump(vlosses, f)
    save_models(agents=agents, embedding_stack=embedding_stack, shared_critic=shared_critic, optimizer=material_optimizer, run_number=2, complete_path=complete_path)

def outsourcing_stage(train_data: list[Instance], val_data: list[Instance], agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, substage: int, switch_batch: int, validation_rate: int, device: str, interactive: bool=False, debug_mode: bool=False):
    """
        Third (3rd) optimization stage: outsourcing agent alone
    """
    outsourcing_optimizer = torch.optim.Adam(list(agents[OUTSOURCING][AGENT].parameters()), lr=LEARNING_RATE)
    debug_print: Callable = debug_printer(debug_mode)
    vlosses = MAPPO_Losses(agent_names=[AGENTS[OUTSOURCING]])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _outsourcing_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Outsourcing loss (policy)", title="Outsourcing loss (policy)", color="red", show=interactive)
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % switch_batch == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch: list[Instance] = random.sample(train_data, batch_size)
        train_or_validate_batch(agents, current_batch, train=True, epochs=epochs, optimizer=outsourcing_optimizer, agent_names=[AGENTS[OUTSOURCING]], solve_function=solve_function, device=device, debug=debug_mode, trainable=[1,0,0])
        if iteration % validation_rate == 0:
            debug_print("\t time to validate the loss...")
            agents[OUTSOURCING][AGENT].eval()
            with torch.no_grad():
                current_vloss: MAPPO_Loss = train_or_validate_batch(agents, val_data, train=False, epochs=-1, agent_names=[AGENTS[OUTSOURCING]], optimizer=None, solve_function=solve_function, device=device, debug=debug_mode, trainable=[1,0,0])
                vlosses.add(current_vloss)
                _vloss_TRACKER.update(current_vloss.value_loss)
                _outsourcing_loss_TRACKER.update(current_vloss.get(AGENTS[OUTSOURCING]).policy_loss)
            agents[OUTSOURCING][AGENT].train()
    complete_path = path + directory.models
    _run: int = 2 + substage
    _vloss_TRACKER.save(complete_path+'/stage_'+str(_run)+'_value_loss')
    _outsourcing_loss_TRACKER.save(complete_path+'/stage_'+str(_run)+'_outsourcing_loss')
    with open(complete_path+'/validation_'+str(_run)+'.pkl', 'wb') as f:
        pickle.dump(vlosses, f)
    save_models(agents=agents, embedding_stack=embedding_stack, shared_critic=shared_critic, optimizer=outsourcing_optimizer, run_number=_run, complete_path=complete_path)

def multi_agent_stage(train_data: list[Instance], val_data: list[Instance], agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, solve_function: Callable, path: str, epochs: int, iterations:int, batch_size: int, switch_batch: int, validation_rate: int, device: str, interactive: bool=False, debug_mode: bool=False):
    """
        Last (4th) optimization stage: all three agents together
    """
    multi_agent_optimizer = torch.optim.Adam(list(agents[SCHEDULING][AGENT].parameters()) + list(agents[OUTSOURCING][AGENT].parameters()) + list(agents[MATERIAL_USE][AGENT].parameters()), lr=LEARNING_RATE)
    debug_print: Callable = debug_printer(debug_mode)
    vlosses = MAPPO_Losses(agent_names=[name for _,name in agents])
    _vloss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Value loss", title="Value loss", color="blue", show=interactive)
    _scheduling_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Scheduling loss (policy)", title="Scheduling loss (policy)", color="green", show=interactive)
    _material_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Material use loss (policy)", title="Material use loss (policy)", color="yellow", show=interactive)
    _outsourcing_loss_TRACKER: LossTracker = LossTracker(xlabel="Training epochs", ylabel="Outsourcing loss (policy)", title="Outsourcing loss (policy)", color="red", show=interactive)
    for iteration in range(iterations):
        print(f"PPO iteration: {iteration+1}/{iterations}:")
        if iteration % switch_batch == 0:
            debug_print(f"\t time to sample new batch of size {batch_size}...")
            current_batch: list[Instance] = random.sample(train_data, batch_size)
        train_or_validate_batch(agents, current_batch, train=True, epochs=epochs, optimizer=multi_agent_optimizer, agent_names=[name for _,name in agents], solve_function=solve_function, device=device, debug=debug_mode, trainable=[1,1,1])
        if iteration % validation_rate == 0:
            debug_print("\t time to validate the loss...")
            for agent,_ in agents:
                agent.eval()
            with torch.no_grad():
                current_vloss: MAPPO_Loss = train_or_validate_batch(agents, val_data, train=False, epochs=-1, agent_names=[name for _,name in agents], optimizer=None, solve_function=solve_function, device=device, debug=debug_mode, trainable=[1,1,1])
                vlosses.add(current_vloss)
                _vloss_TRACKER.update(current_vloss.value_loss)
                _material_loss_TRACKER.update(current_vloss.get(AGENTS[MATERIAL_USE]).policy_loss)
                _scheduling_loss_TRACKER.update(current_vloss.get(AGENTS[SCHEDULING]).policy_loss)
                _outsourcing_loss_TRACKER.update(current_vloss.get(AGENTS[OUTSOURCING]).policy_loss)
            for agent,_ in agents:
                agent.train()
    complete_path = path + directory.models
    _vloss_TRACKER.save(complete_path+'/final_value_loss')
    _scheduling_loss_TRACKER.save(complete_path+'/final_scheduling_loss')
    _outsourcing_loss_TRACKER.save(complete_path+'/final_outsourcing_loss')
    _material_loss_TRACKER.save(complete_path+'/final_material_loss')
    with open(complete_path+'/validation_'+str(5)+'.pkl', 'wb') as f:
        pickle.dump(vlosses, f)
    save_models(agents=agents, embedding_stack=embedding_stack, shared_critic=shared_critic, optimizer=multi_agent_optimizer, run_number=5, complete_path=complete_path)

def multi_stage_pre_train(agents: list[(Module, str)], embedding_stack: Module, shared_critic: Module, path: str, solve_function: Callable, device: str, run_number:int, interactive: bool, debug_mode: bool=False):
    """
        Multi-stage PPO function to pre-train agents on several instances
    """
    _itrs: int = PPO_CONF['train_iterations']
    _epochs: int = PPO_CONF['opt_epochs']
    _batch_size: int = PPO_CONF['batch_size']
    _switch_batch: int = PPO_CONF['switch_batch']
    _validation_rate: int = PPO_CONF['validation_rate']
    _start_time = systime.time()
    print("Loading dataset....")
    instances: list[Instance] = load_training_dataset(path=path, train=True, debug_mode=debug_mode)
    print(f"Dataset loaded after {(systime.time()-_start_time)} seconds!")
    random.shuffle(instances)
    num_val = PPO_CONF['validation']
    train_data, val_data = instances[num_val:], instances[:num_val]
    embedding_stack.train()
    shared_critic.train()
    for agent,_ in agents:
        agent.train()
    
    if run_number <=1:
        print("I. TRAINING STAGE 1: scheduling agent")
        freeze_several(agents, [AGENTS[OUTSOURCING], AGENTS[MATERIAL_USE]])
        agents = scheduling_stage(train_data=train_data,
                                  val_data=val_data,
                                  agents=agents, 
                                  solve_function=solve_function, 
                                  path=path, 
                                  epochs=_epochs, 
                                  iterations=_itrs[0], 
                                  batch_size=_batch_size, 
                                  switch_batch=_switch_batch,
                                  validation_rate=_validation_rate,
                                  device=device,
                                  interactive=interactive, 
                                  debug_mode=debug_mode)
    
    if run_number <=2:
        print("II. TRAINING STAGE 2: material use agent with freezed embedding and critic layers")
        unfreeze_all(agents)
        freeze_several(agents, [AGENTS[SCHEDULING], AGENTS[OUTSOURCING]])
        freeze(embedding_stack)
        freeze(shared_critic)
        agents = material_use_stage(train_data=train_data,
                                    val_data=val_data, 
                                    agents=agents, 
                                    solve_function=solve_function, 
                                    path=path, 
                                    epochs=_epochs, 
                                    iterations=_itrs[1], 
                                    batch_size=_batch_size, 
                                    switch_batch=_switch_batch,
                                    validation_rate=_validation_rate,
                                    device=device, 
                                    interactive=interactive, 
                                    debug_mode=debug_mode)
    
    if run_number <=3:
        print("III. TRAINING STAGE 3.1: outsourcing agent with freezed embedding and critic layers")
        unfreeze_all(agents)
        freeze_several(agents, [AGENTS[SCHEDULING], AGENTS[MATERIAL_USE]])
        freeze(embedding_stack)
        freeze(shared_critic)
        agents = outsourcing_stage(train_data=train_data,
                                   val_data=val_data,
                                   agents=agents, 
                                   solve_function=solve_function, 
                                   path=path, epochs=_epochs, 
                                   iterations=_itrs[2], 
                                   batch_size=_batch_size, 
                                   switch_batch=_switch_batch,
                                   validation_rate=_validation_rate,
                                   device=device, 
                                   substage=1, 
                                   interactive=interactive, 
                                   debug_mode=debug_mode)
    
    if run_number <=4:
        print("III. TRAINING STAGE 3.2: outsourcing agent and all layers")
        unfreeze(embedding_stack)
        agents = outsourcing_stage(train_data=train_data,
                                   val_data=val_data,
                                   agents=agents, 
                                   solve_function=solve_function, 
                                   path=path, epochs=_epochs, 
                                   iterations=_itrs[3], 
                                   batch_size=_batch_size, 
                                   switch_batch=_switch_batch,
                                   validation_rate=_validation_rate,
                                   device=device, 
                                   substage=2,
                                   interactive=interactive, 
                                   debug_mode=debug_mode)

    if run_number <=5:
        print("IV. TRAINING STAGE 4: multi-agent with all layers")
        unfreeze_all(agents)
        unfreeze(embedding_stack)
        unfreeze(shared_critic)
        agents = multi_agent_stage(train_data=train_data,
                                   val_data=val_data,
                                   agents=agents, 
                                   solve_function=solve_function, 
                                   path=path, epochs=_epochs, 
                                   iterations=_itrs[4],
                                   batch_size=_batch_size, 
                                   switch_batch=_switch_batch,
                                   validation_rate=_validation_rate,
                                   device=device, 
                                   interactive=interactive, 
                                   debug_mode=debug_mode)