import gym
import numpy as np
import torch
import torch.nn as nn
import copy
from matplotlib import pyplot as plt
import timeit

def rollout(env, q, ds, eps=0, T=200):

	state = env.reset()
	for t in range(T):
		u = q.control(torch.from_numpy(state).float().unsqueeze(0),
					  eps=eps)
		# u = u.int().numpy().squeeze()

		next_state,r,d,info = env.step(u)
		data = dict(x=state,xp=next_state,r=r,u=u,d=d,info=info)
		state = next_state
		ds.append(data)
		if d:
			break
	return ds

class q_t(nn.Module):
	def __init__(s, xdim, udim, hdim=16):
		super().__init__()
		s.xdim, s.udim = xdim, udim
		s.m = nn.Sequential(
							nn.Linear(xdim, hdim),
							nn.ReLU(True),
							nn.Linear(hdim,int(hdim / 2)),
							nn.ReLU(True),
							nn.Linear(int(hdim / 2), udim),
							)
	def forward(s, x):
		return s.m(x)

	def control(s, state, eps=0):
		# 1. get q values for all controls
		q = s.m(state).detach().numpy()

		# eps-greedy strategy to choose control input
		# note that for eps=0
		# you should return the correct control u
		choice = np.random.rand(1,1)
		if choice <= eps:
			return np.random.choice(2)
		else:
			if q.shape[0] == 1:
				return np.argmax(q)

			return np.argmax(q,axis = 1)
		pass

def loss(q, q_old, ds):
	# 1. sample mini-batch from datset ds
	batch_size = 100
	gamma = 1
	indices = np.random.randint(low = 0,high = len(ds),size = batch_size)

	# tic=timeit.default_timer()

	s = np.array([ds[i]['x'] for i in indices])
	# print("s",s.shape)
	sp = np.array([ds[i]['xp'] for i in indices])
	# print("sp",sp.shape)

	r = np.array([ds[i]['r'] for i in indices])
	# print("r",r.shape)

	a = np.array([ds[i]['u'] for i in indices])
	# print("a",a.shape)

	d = np.array([ds[i]['d'] for i in indices])

	# toc=timeit.default_timer()
	# print("For loops time: ",toc - tic)

	# 2. code up DQN with double-q trick
	new_q_values = q.m(torch.from_numpy(s).float())
	new_q = torch.gather(new_q_values,1,torch.as_tensor(a).unsqueeze(1).long())
	# new_q = torch.zeros((100,1))
	# for i in range(new_q_values.shape[0]):
	# 	new_q[i] = new_q_values[i,a[i]]

	ap = q.control(torch.from_numpy(sp).float())

	old_q_values = q_old.m(torch.from_numpy(sp).float())
	old_q = torch.gather(old_q_values,1,torch.as_tensor(ap).unsqueeze(1).long())

	# old_q = torch.zeros((100,1))
	# for i in range(old_q_values.shape[0]):
	# 	old_q[i] = old_q_values[i,ap[i]]

	old_q[d] = 0

	r = torch.as_tensor(r).unsqueeze(1)

	target = r + gamma * old_q

	loss = (new_q - target.detach()) ** 2
	f = loss.sum()/batch_size
	# print(f)

	# 3. return the objective f
	return f

def evaluate(q,eps = 0):
	# 1. create a new environment e
	test_env = gym.make('CartPole-v0')
	# 2. run the learnt q network for 100 trajectories on
	# this new environment and report the average undiscounted
	# return of these 100 trajectories
	reward_sum = 0
	for i in range(100):
		state = test_env.reset()
		for t in range(200):
			u = q.control(torch.from_numpy(state).float().unsqueeze(0),eps)
			# u = u.int().numpy().squeeze()

			next_state,r,d,info = test_env.step(u)
			# if (i + 1) % 100 == 0:
			# 	test_env.render()
			reward_sum += r
			state = next_state
			if d:
				break

	return reward_sum / 100

if __name__=='__main__':
	env = gym.make('CartPole-v0')

	xdim, udim =    env.observation_space.shape[0], \
					env.action_space.n

	# print("xdim",xdim)
	# print("udim",udim)

	q = q_t(xdim, udim, 32)
	optim = torch.optim.Adam(q.parameters(), lr=5e-4,
						  weight_decay=1e-4)

	ds = []

	# collect few random trajectories with
	# eps=1
	for i in range(100):
		ds = rollout(e, q, ds, eps=1, T=200)
	# print(len(ds))

	eps = 1
	q_old = copy.deepcopy(q)
	n_iterations = 5000
	reward_sum = np.zeros(n_iterations)
	train_reward_sum = np.zeros(int((n_iterations / 1000) + 1))
	for i in range(n_iterations):
		q.train()
		if (i + 1) % 200 == 0:
			q_old = copy.deepcopy(q)
			# for name, param in q_old.named_parameters():
			#     if param.requires_grad:
			#         print(name, param.data)
			# for name, param in q.named_parameters():
			#     if param.requires_grad:
			#         print(name, param.data)

			eps /= 1.05
			print(i + 1)
			print("eps: ",eps)
		# ds = rollout(e, q, ds, eps)
		tic1 = timeit.default_timer()

		state = env.reset()
		for t in range(200):
			u = q.control(torch.from_numpy(state).float().unsqueeze(0),
						  eps=eps)
			# u = u.int().numpy().squeeze()

			next_state,r,d,info = env.step(u)
			data = dict(x=state,xp=next_state,r=r,u=u,d=d,info=info)
			state = next_state
			ds.append(data)
			if d:
				break

		# toc1 = timeit.default_timer()
		# print("Episode generation time: ",toc1 - tic1)

		# perform sgd updates on the q network
		# need to call zero grad on q function
		# to clear the gradient buffer
		q.zero_grad()
		f = loss(q, q_old, ds)

		# tic2 = timeit.default_timer()
		f.backward()
		# toc2 = timeit.default_timer()
		# print("Gradient Update time: ",toc2 - tic2)

		optim.step()

		q.eval()
		reward_sum[i] = evaluate(q)

		if (i + 1) % 100 == 0:
			print(reward_sum[i])

		if (i + 1) % 1000 == 0 or i == 0:
			train_reward_sum[int((i + 1) / 1000)] = evaluate(q,eps)
		# print('Logging data to plot')

plt.figure(1)
plt.title("Average Reward vs Number of Parameter Updates \n Learning Rate = 5e-4")
plt.xlabel("Number of iterations")
plt.ylabel("Average reward over 100 trajectories")
plt.plot(reward_sum)

plt.figure(2)
plt.title("Average Evaluation Environment Rewards every 1000 parameter updates \n Learning Rate = 5e-4")
plt.xlabel("Number of Parameter Updates")
plt.ylabel("Average reward over 100 trajectories")
indices = np.arange(int((n_iterations / 1000) + 1)) * 1000
indices[1:] -= 1
plt.plot(indices,reward_sum[indices])

plt.figure(3)
plt.title("Average Training Environment Rewards every 1000 parameter updates \n Learning Rate = 5e-4")
plt.xlabel("Number of Parameter Updates")
plt.ylabel("Average reward over 100 trajectories")
plt.plot(indices,train_reward_sum)

plt.show()