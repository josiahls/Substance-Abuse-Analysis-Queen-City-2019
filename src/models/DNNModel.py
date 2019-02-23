import sys

from torch.utils.data.dataset import random_split

sys.path.append('../')

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.data.SubstanceAbuseDataset import SubstanceAbuseDataset
from src.data.Transforms import *
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
print(f'Using torch version {torch.__version__}')

# Create random Tensors to hold inputs and outputs
dataset = SubstanceAbuseDataset('HackTrain.csv', './', Compose([ToXY(), ToTensor()]), n_rows=100)
test_set_size = .2
train_dataset, validation_dataset = random_split(dataset, lengths=[int(len(dataset) * test_set_size),
                                                                   int(len(dataset) - int(len(dataset)
                                                                                    * test_set_size))])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=True, num_workers=4)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
H = 200
model = torch.nn.Sequential(
    torch.nn.Linear(len(train_dataset[0]['X']), H),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H, len(train_dataset[0]['Y'])),
)

# model = torch.nn.Sequential( torch.nn.Linear(10, 20), torch.nn.Linear(20, 2))
# Try loading cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model.to(device=device)

# Setup the writer
writer = SummaryWriter()

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-3
for t in range(500):
    cum_loss = []
    for i_batch, sample_batched in enumerate(train_loader):
        x_batch = sample_batched['X'].to(device=device)
        y_batch = sample_batched['Y'].to(device=device)
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x_batch)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = torch.sqrt(loss_fn(y_pred, y_batch))

        cum_loss.append(loss.cpu())
        # data grouping by `slash`

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    cum_test_loss = []
    for i_batch, sample_batched in enumerate(validation_loader):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(sample_batched['X'].float())

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = torch.sqrt(loss_fn(y_pred, sample_batched['Y'].float()))
        cum_test_loss.append(loss.cpu())

    print(f'Train lost: {torch.mean(torch.from_numpy(np.array(cum_loss, dtype=np.float64)))} Test Lost: {torch.mean(torch.from_numpy(np.array(cum_test_loss, dtype=np.float64)))}')
    writer.add_scalar('data/train', torch.mean(torch.from_numpy(np.array(cum_loss, dtype=np.float64))), t)
    writer.add_scalar('data/test', torch.mean(torch.from_numpy(np.array(cum_test_loss, dtype=np.float64))), t)

    # plotter.live_plotter(torch.mean(torch.from_numpy(np.array(cum_test_loss, dtype=np.float64))))

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
