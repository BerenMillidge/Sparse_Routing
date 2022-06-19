# A quick demonstration of how pc inference can mimic capsule routing with L1 norm to enforce sparsity
# here we setup a simple hardcoded PCN and only do inference on a single layer. We imagine the 'inputs' are from capsules at a lower layer and the goal is to route the inputs to the best capsule to process them using iterative inference
# we find that standard PC inference will tend to distribute activity between the capsules, pc inference with L1 norm is highly sparse and tends to put most activity to only one capsule (as intended)
# pc inference with L2 norm distributes activity almost equally between capsules so does anti-routing
# These effects are strongest when only output loss is considered since the middle losses of the network favour keeping the capsule values close to their prior values, which is not highly sparse, but the same effects still apply
import torch
from torch import nn, optim
import torch.autograd.functional as taf
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# setup loss functions
def standard_pc_error(e,x):
  return torch.sum(torch.square(e))

def pc_plus_l2(e, x):
  return standard_pc_error(e,x) + torch.sum(torch.square(x))

def pc_plus_l1(e,x):
  return standard_pc_error(e,x) + torch.sum(torch.abs(x))


# setup network and parameters -- weights and inputs and targets are fixed to fairly arbitrary values
inputs = torch.tensor([-1,5,2]).reshape(3,1).float()
target = torch.tensor([2]).reshape(1,1).float()
W1 = torch.eye(3).float()
W2 = torch.tensor([1,1,1]).reshape(1,3).float()
print(inputs.shape)
print(target.shape)
print(W1.shape)
print(W2.shape)
lr = 1e-2
momentum = 0.0

x2 = nn.Parameter(W1 @ inputs)
print(x2.shape)
opt = optim.SGD([x2], lr=lr, momentum=momentum)
opt.zero_grad()

# we initialize the network and use pytorch's autodiff to optimize the PCN since the point here is routing and not so much the biological plausibility of the updates, although both L1 and L2 regularization are not hard
# to implement either
def run_pc_inference(N_steps, opt,x2,W1,W2, inputs, targets, loss_fn, only_output_loss=False):
  x2s = []
  e1s = []
  e2s = []
  for i in range(N_steps):
    opt.zero_grad()
    e1 = x2 - (W1 @ inputs)
    e2 = (W2 @ x2) - targets
    if only_output_loss:
      loss = loss_fn(e2,x2)
    else:
      loss =  loss_fn(e2,x2) +  loss_fn(e1,x2)
    loss.backward()
    opt.step()
    x2s.append(deepcopy(x2.detach().reshape(3,).numpy()))
    e1s.append(deepcopy(loss_fn(e1,x2).detach().numpy()))
    e2s.append(deepcopy(loss_fn(e2,x2).detach().numpy()))
  return x2s, e1s,e2s


def make_plots(N_steps, opt, x2, W1, W2, inputs, targets, loss_fn, only_output_loss = False):
  # do experiment and setup output
  x2s, e1s, e2s = run_pc_inference(N_steps, opt, x2, W1, W2, inputs, targets, loss_fn = loss_fn, only_output_loss = only_output_loss)
  print(len(x2s))
  x2s = np.array(x2s)
  e1s = np.array(e1s)
  e2s = np.array(e2s)
  print(x2s.shape)

  sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
  # create savename automatically
  loss_fn_name = str(loss_fn.__name__) # a bit of a hack this
  save_string = "_" + loss_fn_name + "_"
  if only_output_loss:
    save_string += "only_output_loss"
  save_string += ".jpg"
  
  # evolution of activities
  fig = plt.figure(figsize=(12,10))
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.title("Activity Evolution " + loss_fn_name,fontsize=30)
  plt.plot(x2s)
  plt.xlabel("Inference Timestep",fontsize=28)
  plt.ylabel("Activity Value",fontsize=28)
  plt.yticks(fontsize=20)
  plt.xticks(fontsize=20)
  #plt.legend(fontsize=25)
  fig.tight_layout()
  plt.savefig("activity_evolution" + save_string, format="jpeg")
  plt.show()

  # evolution of middle loss (e1)
  fig = plt.figure(figsize=(12,10))
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.title("Middle Loss Evolution " + loss_fn_name,fontsize=30)
  plt.plot(e1s)
  plt.xlabel("Inference Timestep",fontsize=28)
  plt.ylabel("Loss Value",fontsize=28)
  plt.yticks(fontsize=20)
  plt.xticks(fontsize=20)
  #plt.legend(fontsize=25)
  fig.tight_layout()
  plt.savefig("middle_loss_evolution" + save_string, format="jpeg")
  plt.show()

  # evolution of output loss (e2)
  fig = plt.figure(figsize=(12,10))
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.title("Output Loss Evolution " + loss_fn_name,fontsize=30)
  plt.plot(e2s)
  plt.xlabel("Inference Timestep",fontsize=28)
  plt.ylabel("Loss Value",fontsize=28)
  plt.yticks(fontsize=25)
  plt.xticks(fontsize=20)
  #plt.legend(fontsize=25)
  fig.tight_layout()
  plt.savefig("output_loss_evolution" + save_string, format="jpeg")
  plt.show()

  # activity values after inference (indicator of sparsity / routing)
  fig = plt.figure(figsize=(12,10))
  plt.title("Equilibrium Activity Values " + loss_fn_name,fontsize=30)
  plt.bar([1,2,3],x2s[-1],width=0.8, align='center')
  plt.xlabel("Activity Dimension",fontsize=28)
  plt.ylabel("Activity Value",fontsize=28)
  plt.yticks(fontsize=20)
  plt.xticks([1,2,3],fontsize=20)
  #plt.legend(fontsize=25)
  fig.tight_layout()
  plt.savefig("equilibrium_bar_evolution" + save_string, format="jpeg")
  plt.show()

if __name__ == '__main__':
    make_plots(500, opt, x2, W1, W2, inputs, target, standard_pc_error, only_output_loss = True)
    make_plots(500, opt, x2, W1, W2, inputs, target, pc_plus_l1, only_output_loss = True)
    make_plots(500, opt, x2, W1, W2, inputs, target, pc_plus_l2, only_output_loss = True)
    make_plots(500, opt, x2, W1, W2, inputs, target, standard_pc_error, only_output_loss = False)
    make_plots(500, opt, x2, W1, W2, inputs, target, pc_plus_l1, only_output_loss = False)
    make_plots(500, opt, x2, W1, W2, inputs, target, pc_plus_l2, only_output_loss = False)