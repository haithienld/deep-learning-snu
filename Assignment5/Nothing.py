import gym #1
import time #1 
from collections import deque #3
import numpy as np #3 #5
import tensorflow as tf #2
import numpy as np #4
import matplotlib.pyplot as plt #5
import os
#1. Create Environment 

def create_cart_pole_env(env):
    env.reset()
    rewards = []
    tic = time.time()
    for _ in range(1000):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        rewards.append(reward)
        if done:
            rewards = []
            env.reset()
    toc = time.time()
    if toc-tic > 10:
        env.close()
    print(rewards)
#2 Create DQNetwork Implements the Deep Neural Network

class DQNetwork:
    def __init__(self,\
                 learning_rate=0.01, \
                 state_size=4,\
                 action_size=2, \
                 hidden_size=10,\
                 name='DQNetwork'):

         with tf.variable_scope(name):
            self.inputs_ = \
                         tf.placeholder\
                         (tf.float32,\
                          [None, state_size],\
                          name='inputs')
            
            self.actions_ = tf.placeholder\
                            (tf.int32,[None],\
                             name='actions')
            
            one_hot_actions =tf.one_hot\
                              (self.actions_,\
                               action_size)
            
            self.targetQs_ = tf.placeholder\
                             (tf.float32,[None],\
                              name='target')
            
            self.fc1 =tf.contrib.layers.fully_connected\
                       (self.inputs_,\
                        hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected\
                       (self.fc1,\
                        hidden_size)

            self.output = tf.contrib.layers.fully_connected\
                          (self.fc2,\
                           action_size,activation_fn=None)

            self.Q = tf.reduce_sum(tf.multiply\
                                   (self.output,\
                                    one_hot_actions),\
                                   axis=1)
            
            self.loss = tf.reduce_mean\
                        (tf.square(self.targetQs_ - self.Q))
            
            self.opt = tf.train.AdamOptimizer\
                       (learning_rate).minimize(self.loss)
#3 Create ReplayMemory Implements the experience replay method

class replayMemory():
    def __init__(self, max_size = 1000):
        self.buffer = \
                    deque(maxlen=max_size)
    
    def build(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice\
              (np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


#4 Solves the Cart-Pole environment with the trained neural network - Agent

def solve_cart_pole(env,dQN,state,sess):
    test_episodes = 10
    test_max_steps = 400
    env.reset()
    for ep in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render() 
        
            # Get action from Q-network
            Qs = sess.run(dQN.output, \
                          feed_dict={dQN.inputs_: state.reshape\
                                     ((1, *state.shape))})
            action = np.argmax(Qs)
        
            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
        
            if done:
                t = test_max_steps
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                t += 1     
#5 Plots the final rewards versus the episodes
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def plot_result(rew_list):
    eps, rews = np.array(rew_list).T
    smoothed_rews = running_mean(rews, 10)
    smoothed_rews = running_mean(rews, 10)  
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    
#6 The main program
env = gym.make('CartPole-v0')
create_cart_pole_env(env)


#Build the deep neural network
tf.reset_default_graph()
deepQN = DQNetwork(name='main', \
                  hidden_size=64, \
                  learning_rate=0.0001)

# Initialize the simulation
env.reset()

# Take one random step to get the pole and cart moving
state, rew, done, _ = env.step(env.action_space.sample())
memory = replayMemory(max_size=10000)

# Make a bunch of random actions and store the experiences
pretrain_length= 20

for j in range(pretrain_length):
    action = env.action_space.sample()
    next_state, rew, done, _ = \
                env.step(env.action_space.sample())
    if done:
        env.reset()
        memory.build((state, action, rew, np.zeros(state.shape)))
        state, rew, done, _ = \
               env.step(env.action_space.sample())
    else:
        memory.build((state, action, rew, next_state))
        state = next_state
        

# Exploration parameters
# exploration probability at start
start_exp = 1.0
# minimum exploration probability 
stop_exp = 0.01
# exponential decay rate for exploration prob
decay_rate = 0.0001            

# Train the DQN with new experiences
rew_list = []
train_episodes = 10000
max_steps=200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for ep in range(1, train_episodes):
        tot_rew = 0
        t = 0
        while t < max_steps:
            step += 1
            explore_p = stop_exp + \
                        (start_exp - stop_exp)*\
                        np.exp(-decay_rate*step)
            
            if explore_p > np.random.rand():
                action = env.action_space.sample()
                
            else:
                Qs = sess.run(deepQN.output, \
                              feed_dict={deepQN.inputs_: \
                                         state.reshape\
                                         ((1, *state.shape))})
                action = np.argmax(Qs)

            next_state, rew, done, _ = env.step(action)
            tot_rew += rew
            
            if done:
                next_state = np.zeros(state.shape)
                t = max_steps
               
                print('Episode: {}'.format(ep),
                      'Total rew: {}'.format(tot_rew),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))
                
                rew_list.append((ep, tot_rew))
                memory.build((state, action, rew, next_state))
                env.reset()
                state, rew, done, _ = env.step\
                                         (env.action_space.sample())

            else:
                memory.build((state, action, rew, next_state))
                state = next_state
                t += 1

            batch_size = pretrain_length               
            states = np.array([item[0] for item in memory.sample(batch_size)])
            actions = np.array([item[1] for item in memory.sample(batch_size)])
            rews = np.array([item[2] for item in memory.sample(batch_size)])
            next_states = np.array([item[3] for item in memory.sample(batch_size)])
            
            target_Qs = sess.run(deepQN.output, \
                                 feed_dict=\
                                 {deepQN.inputs_: next_states})

            target_Qs[(next_states == \
                       np.zeros(states[0].shape))\
                      .all(axis=1)] = (0, 0)
            
            targets = rews + 0.99 * np.max(target_Qs, axis=1)

            loss, _ = sess.run([deepQN.loss, deepQN.opt],
                                feed_dict={deepQN.inputs_: states,
                                           deepQN.targetQs_: targets,
                                           deepQN.actions_: actions})

    env = gym.make('CartPole-v0')
    solve_cart_pole(env,deepQN,state,sess)
    plot_result(rew_list)
