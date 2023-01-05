from utils import *
from torch import optim
import torch.nn.functional as F
import random

'''Learner file (BC + DAgger)'''

class BC:
    def __init__(self, net, loss_fn, data_size, batch_size=128):
        self.net = net
        self.loss_fn = loss_fn
        
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.data_size = data_size
        self.losses = []

    def get_batch(self):
        indices = random.sample(range(self.data_size), self.batch_size)
        batch_states  = self.states[indices]
        batch_actions = self.actions[indices]

        return batch_states, batch_actions
        
    def learn(self, env, states, actions, n_steps=1e4, every=100,truncate=False, show=True, get_losses = True):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).

        '''
         env - use it of evaluate policy (e.g. whatever you want, rollout) ... 
         states - input, actions - output 
         n_steps - the number of batches of the states (sample from states)
         truncate - remove the last two parts of the state for causal confounds 
        '''
        n_steps = int(n_steps)
        
        self.states   = torch.from_numpy(states).float()
        self.actions  = torch.from_numpy(actions).float()

        running_loss = 0.0

        losses = []

        if every<n_steps:
            every = n_steps
        for i in range(n_steps):
             # get the inputs; data is a list of [inputs, labels]
            inputs, labels = self.get_batch()
            # zero the parameter gradients
            self.opt.zero_grad()
            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels.long().squeeze())
            loss.backward()
            self.opt.step()
            # print statistics
            running_loss += loss.item()
            # log loss
            if i % every == every-1 and get_losses: 
                if show:
                    print(f'[{i + 1}] loss: {running_loss/every}')
                losses.append([running_loss])
                running_loss = 0.0

        return losses

    def learn_more(self, n_steps=50, every=100):
        ''' n_steps - the number of batches of the states (sample from states)
        '''
        n_steps = int(n_steps)
        running_loss = 0.0
        losses = []
        if every>n_steps:
            every = n_steps

        for i in range(n_steps):
            inputs, labels = self.get_batch()
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels.long().squeeze())
            loss.backward()
            self.opt.step()

            running_loss += loss.item()
            # log loss
            if i % every == every-1: 
                # print("--- learning more... i is   :", i)     
                # print("--- learning more... loss is:", running_loss)     
                losses.append([running_loss])
                running_loss = 0.0
            
        
        return losses



class DAgger:
    def __init__(self, net, loss_fn, expert):
        self.net = net
        self.loss_fn = loss_fn
        self.expert = expert
        
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.losses = []
        self.performances = []
    def generate_data(self, X=25):
        states  = None
        actions = None
        for i in range(X):
            current_states, current_actions = expert_rollout(self.expert, self.env, truncate=False)
            
            if states is None and actions is None:
                states  = np.array(current_states, dtype='float')
                actions = np.array(current_actions, dtype='float')
            else:
                states  = np.vstack((states,current_states))
                actions = np.vstack((actions, current_actions))
        return states, actions

    def interact_with_expert(self):
        # rollout policy
        states, _ = rollout(self.net.net, self.env, truncate=False)
        states  = np.array(states, dtype='float')
        actions_star = []
        
        for i in range(states.shape[0]):
            action = self.expert.predict(states[i])[0]
            actions_star.append([action])


        actions_star = np.array(actions_star, dtype='float')


        self.states  = np.vstack((self.states, states))
        self.actions = np.vstack((self.actions,actions_star))

    def performance(self,i):
        policy = argmax_policy(self.net.net)
        performance = get_policy_performance(policy,self.env)
        print(f"BC iteraration:{i} with performance:{performance}")

        self.performances.append([i,performance])


    def learn(self, env, n_steps=4e3, truncate=False, N=15):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).
        # Make sure you are making the learning process fundamentally expert-interactive.
        self.n_steps = n_steps
        self.env = env
        self.states, self.actions = self.generate_data()
        
        data_size = self.states.shape[0]
        # bc = BC(net, loss_fn,data_size,batch_size)

        self.net = BC(self.net, self.loss_fn,data_size)

        
        for i in range(N):
            print("DAgger iteration:", i)
            training_losses = self.net.learn(env, self.states, self.actions, self.n_steps,every=self.n_steps,show=False)
            self.losses.append(training_losses[-1])
            self.performance(i)
            self.interact_with_expert()


        return self.performances, self.losses




