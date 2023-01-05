import gym
from stable_baselines3.ppo import PPO
import torch.nn as nn
import argparse
import numpy as np

import time
from datetime import datetime

from learners import *
from utils import *

def make_env():
    return gym.make("LunarLander-v2")

def get_expert():
    return PPO.load("./experts/LunarLander-v2/lunarlander_expert")

def get_expert_performance(env, expert):
    Js = []
    for _ in range(100):
        obs = env.reset()
        J = 0
        done = False
        hs = []
        while not done:
            action, _ = expert.predict(obs)
            obs, reward, done, info = env.step(action)
            hs.append(obs[1])
            J += reward
        Js.append(J)
    ll_expert_performance = np.mean(Js)
    return ll_expert_performance


def render_agent(env,policy,num_steps = 1500):
    env.render(mode = "human")

    # Number of steps you run the agent for 
    num_steps = 1500
    obs = env.reset()

    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 
        # action = env.action_space.sample()



        action = policy(obs)
    
        # detach action and convert to np array
        if isinstance(action, torch.FloatTensor) or isinstance(action, torch.Tensor):
            action = action.detach().numpy()


        
        # apply the action
        obs, reward, done, info = env.step(action)
        
        # Render the env
        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)
        
        # If the epsiode is up, then start another one
        if done:
            env.reset()


def BC_performances(env, num_iterations, net, loss_fn, states, actions, batch_size =128):
    performances = []
    losses = []
    data_size = states.shape[0]
    print("data_size is: ", data_size)
    bc = BC(net, loss_fn,data_size,batch_size)
    total = np.sum(num_iterations)
    training_losses = None
    for i, n in enumerate(num_iterations):

        print("Iteration", i)
        # training 
        print("--- training ...")
        if i == 0:
            training_losses = bc.learn(env, states, actions,n_steps=n, show=False)        
        else:       
            training_losses = bc.learn_more(n_steps=n)

        losses.extend(training_losses[-1])

        # Performance 
        print("--- computing bc performance for "+str(np.sum(num_iterations[:i+1]))+"/"+str(total)+" iterations")
        policy = argmax_policy(bc.net)

        p = get_policy_performance(policy, env)
        print("--- performace is --> ", p)

        # save performance 
        performances.append(p)

        

    return performances, losses



def BC_compare_training_iterations(env,loss_fn, expert, num_iterations,  X, truncate):
    """
    How well does the BC argmax_policy perform when rolled out (total rewards)? 

    As you increase the number of training iterations of BC,
         what happens to train / validation loss / total reward?
    """
    
    net = None
    if truncate:
        net = create_net(input_dim=6, output_dim=4)
    else:
        net = create_net(input_dim=8, output_dim=4)

    
    states  = None
    actions = None
    for i in range(X):
        current_states, current_actions = expert_rollout(expert, env, truncate=False)
        
        if states is None and actions is None:
            states  = np.array(current_states, dtype='float')
            actions = np.array(current_actions, dtype='float')
        else:
            states  = np.vstack((states,current_states))
            actions = np.vstack((actions, current_actions))


    performances, losses =  BC_performances(env, num_iterations, net, loss_fn, states, actions, batch_size=100)

        
    return performances, losses


def BC_compare_iter(env,loss_fn, expert,start,stop,space, X, truncate=False):
    
    iterations = np.linspace(start,stop,space,dtype=int)
    num_iterations = np.zeros(iterations.shape)
    num_iterations[0] = iterations[0]
    num_iterations[1:] = iterations[1:] - iterations[:-1]
    performances, losses  = BC_compare_training_iterations(env,loss_fn, expert, num_iterations.astype(int), X, truncate)

    performances = np.asarray(performances)
    losses = np.asarray(losses)
    print("iterations:   ", iterations)
    print("performances: ", performances)
    print("losses:       ", losses)

    data = np.vstack((iterations,performances, losses))

    part = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    np.savetxt("BC_compare_iter"+part+".csv", data.T, delimiter=",",fmt="%10.5f")

def DAgger_compare_training_iterations():
    """
    Some easy extensions that we can check is to see how the size of the dataset plays a role in BC performance. Ideally, the larger the dataset, the more signal BC will have to train on, so hopefully we can see better performance!
    """

    pass


def BC_compare_dataset_size(net, env, loss_fn, expert):
    """
    How well does the BC argmax_policy perform when rolled out (total rewards)? 
    As you increase the number of training iterations of BC,
    what happens to train / validation loss / total reward?
    """
    X = 25 
    states  = None
    actions = None

    data = []
    for i in range(X+10):
        current_states, current_actions = expert_rollout(expert, env, truncate=False)
        
        if states is None and actions is None:
            states  = np.array(current_states, dtype='float')
            actions = np.array(current_actions, dtype='float')
        else:
            states  = np.vstack((states,current_states))
            actions = np.vstack((actions, current_actions))


        if i >= 8:
            bc = BC(net, loss_fn,batch_size=100)
            bc.learn(env, states, actions,n_steps=1e4, show=False, get_losses = False)

            policy = argmax_policy(bc.net)
            print(">>Dataset compare at iteration: ", i)
            performance = get_policy_performance(policy, env)
            data.append([i,performance,np.max(states.size)])


    part = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    print(data)
    data = np.asarray(data)

    np.savetxt("BC_compate_data_size"+part+".csv", data, delimiter=",",fmt="%10.5f")

    

def main(args):
    env = make_env()
    expert = get_expert()
    
    performance = get_expert_performance(env, expert)
    print('=' * 20)
    print(f"Expert performance: {performance}")
    print('=' * 20)
    
    # net + loss fn
    if args.truncate:
        net = create_net(input_dim=6, output_dim=4)
    else:
        net = create_net(input_dim=8, output_dim=4)
    
    loss_fn = nn.CrossEntropyLoss()
    

    if args.analytics:
        BC_compare_iter(env,loss_fn, expert,start=100,stop=10000,space=15, X=50, truncate=False)
        # BC_compare_dataset_size(net, env,loss_fn, expert)
    
    elif args.bc:
        # TODO: train BC
        # Things that need to be done:
        # - Roll out the expert for X number of trajectories (a standard amount is 10).
        # get expert data 
        X = 25 
        states  = None
        actions = None
        for i in range(X):
            current_states, current_actions = expert_rollout(expert, env, truncate=False)
            
            if states is None and actions is None:
                states  = np.array(current_states, dtype='float')
                actions = np.array(current_actions, dtype='float')
            else:
                states  = np.vstack((states,current_states))
                actions = np.vstack((actions, current_actions))


        # - Create our BC learner, and train BC on the collected trajectories.
        # - It's up to you how you want to structure your data!
        bc = BC(net, loss_fn,batch_size=100)
        losses = bc.learn(env, states, actions,n_steps=1e4, every=1000)

        losses = np.array(losses)

        part = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        print(losses)
        np.savetxt("BC_training_iterations"+part+".csv", losses, delimiter=",",fmt="%10.5f")
        

        # - Evaluate the argmax_policy by printing the total rewards.
        # bc.net.detach()
        policy = argmax_policy(bc.net)

        N = 1

        performance = 0.0
        for i in range(N):
            print("Computing BC Performance -- iteration:", i)
            performance = performance + eval_policy(policy, env, truncate=False)
        performance = performance/float(N)

        print(f"BC performance: {performance}")

        # render_agent(env, policy, num_steps = 1500)
        print("End of Behavior Cloning")
    else:
        # Train DAgger
        # Things that need to be done.
        # - Create our DAgger learner.
        # - Set up the training loop. Make sure it is fundamentally interactive!
        # - It's up to you how you want to structure your data!
        # - Evaluate the argmax_policy by printing the total rewards.
        print("Doing DAgger")
        dg = DAgger(net, loss_fn, expert)
        n=15
        performances, losses = dg.learn(env,N=n)
        print(performances)
        print(losses)
        
        losses = np.array(losses)
        performances = np.array(performances)

        print("performances: ", performances)
        print("losses:       ", losses)
        
        data = np.hstack((performances, losses))
        part = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        np.savetxt("DAgger_iterations_performance_losses_N_"+str(n)+"_"+part+".csv", data, delimiter=",",fmt="%10.5f")

        print("End of DAgger")
        
def get_args():
    parser = argparse.ArgumentParser(description='imitation')
    parser.add_argument('--bc', action='store_true', help='whether to train BC or DAgger')
    parser.add_argument('--analytics', action='store_true', help='whether to train BC or DAgger')
    parser.add_argument('--n_steps', type=int, default=10000, help='number of steps to train learner')
    parser.add_argument('--truncate', action='store_true', help='whether to truncate env')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(get_args())
