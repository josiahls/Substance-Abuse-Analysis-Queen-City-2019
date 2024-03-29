from datetime import datetime
from queue import Queue, LifoQueue

from torch import nn


class DNNModel:
    def __init__(self):
        self.train_dataset = None
        self.validation_dataset = None
        self.train_loader = None
        self.validation_loader = None
        self.device = None
        self.model = None
        self.test_dataset = None
        self.train_val_dataset = None

    def load_data(self):
        from torchvision.transforms import Compose
        from src.data.SubstanceAbuseDataset import SubstanceAbuseDataset
        from src.data.Transforms import ToXY, ToTensor
        from torch.utils.data.dataset import random_split
        # Create random Tensors to hold inputs and outputs
        print('Loading Pre Train Set')
        train_val_dataset_pre = SubstanceAbuseDataset('HackTrain.csv', './', Compose([ToXY(), ToTensor()]), n_rows=200000)
        print('Loading Test Set')
        self.test_dataset = SubstanceAbuseDataset('HackTest.csv', './', Compose([ToXY(), ToTensor()]), n_rows=None,
                                                  master_columns=train_val_dataset_pre.substance_abuse_frame.columns)
        print('Loading Train')
        self.train_val_dataset = SubstanceAbuseDataset('HackTrain.csv', './', Compose([ToXY(), ToTensor()]),
                                                       master_columns=self.test_dataset.substance_abuse_frame.columns,
                                                       dataframe=train_val_dataset_pre.raw_frame)

        validation_set_size = .3
        self.train_dataset, self.validation_dataset = random_split(self.train_val_dataset,
                                                                   lengths=[int(len(
                                                                       self.train_val_dataset) * validation_set_size),
                                                                            int(len(self.train_val_dataset) - int(
                                                                                len(self.train_val_dataset)
                                                                                * validation_set_size))])

    def train(self):
        import sys
        sys.path.append('../')
        from tensorboardX import SummaryWriter
        import torch
        import numpy as np
        from torch.utils.data import DataLoader

        self.train_loader = DataLoader(self.train_dataset, batch_size=10, shuffle=True, num_workers=4)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=10, shuffle=True, num_workers=4)

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        print(f'Using torch version {torch.__version__}')

        # Use the nn package to define our self.model as a sequence of layers. nn.Sequential
        # is a Module which contains other Modules, and applies them in sequence to
        # produce its output. Each Linear Module computes output from input using a
        # linear function, and holds internal Tensors for its weight and bias.
        H = 100
        self.model = torch.nn.Sequential(
            torch.nn.Linear(len(self.train_dataset[0]['X']), 41),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(41, 7),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(7, len(self.train_dataset[0]['Y'])),
        )

        # class DropModel(nn.Module):
        #     def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, hidden_dim3, p):
        #         super(DropModel, self).__init__()
        #         self.hidden_dim1 = hidden_dim1
        #         self.hidden_dim2 = hidden_dim2
        #         self.hidden_dim3 = hidden_dim3
        #         self.linear1 = nn.Linear(input_dim, hidden_dim1)
        #         self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        #         self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        #         self.linear4 = nn.Linear(hidden_dim3, output_dim)
        #
        #         self.drop_layer = nn.Dropout(p=p)
        #         self.leaky_relu = nn.LeakyReLU()
        #
        #     def forward(self, input):
        #         out = self.linear1(input)
        #         out = self.leaky_relu(out)
        #         out = self.drop_layer(out)
        #         out = self.linear2(out)
        #         out = self.leaky_relu(out)
        #         out = self.linear3(out)
        #         out = self.leaky_relu(out)
        #         out = self.linear4(out)
        #         return out
        #
        # self.model = DropModel(len(self.train_dataset[0]['X']), len(self.train_dataset[0]['Y']), 9, 6, 4, 0.5)

        # self.model = torch.nn.Sequential( torch.nn.Linear(10, 20), torch.nn.Linear(20, 2))
        # Try loading cuda
        use_cuda = torch.cuda.is_available()
        print(f'Using Cuda? {use_cuda}')
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.model.to(device=self.device)

        # Setup the writer
        writer = SummaryWriter()

        # The nn package also contains definitions of popular loss functions; in this
        # case we will use Mean Squared Error (MSE) as our loss function.
        loss_fn = torch.nn.MSELoss(size_average=False)

        loss_tracking = []
        learning_rate = 1e-3
        for t in range(40):
            learning_rate *= 0.8
            cum_loss = []

            self.model.train()
            for i_batch, sample_batched in enumerate(self.train_loader):
                x_batch = sample_batched['X'].to(device=self.device)
                y_batch = sample_batched['Y'].to(device=self.device)
                # Forward pass: compute predicted y by passing x to the self.model. Module objects
                # override the __call__ operator so you can call them like functions. When
                # doing so you pass a Tensor of input data to the Module and it produces
                # a Tensor of output data.
                y_pred = self.model(x_batch)

                # Compute and print loss. We pass Tensors containing the predicted and true
                # values of y, and the loss function returns a Tensor containing the
                # loss.
                loss = torch.sqrt(loss_fn(y_pred, y_batch))

                cum_loss.append(loss.cpu())
                # data grouping by `slash`

                # Zero the gradients before running the backward pass.
                self.model.zero_grad()

                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the self.model. Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will compute gradients for
                # all learnable parameters in the self.model.
                loss.backward()

                # Update the weights using gradient descent. Each parameter is a Tensor, so
                # we can access its gradients like we did before.
                with torch.no_grad():
                    for param in self.model.parameters():
                        param -= learning_rate * param.grad

            cum_test_loss = []
            self.model.eval()
            for i_batch, sample_batched in enumerate(self.validation_loader):
                x_batch = sample_batched['X'].to(device=self.device)
                y_batch = sample_batched['Y'].to(device=self.device)
                # Forward pass: compute predicted y by passing x to the self.model. Module objects
                # override the __call__ operator so you can call them like functions. When
                # doing so you pass a Tensor of input data to the Module and it produces
                # a Tensor of output data.
                y_pred = self.model(x_batch)

                # Compute and print loss. We pass Tensors containing the predicted and true
                # values of y, and the loss function returns a Tensor containing the
                # loss.
                loss = torch.sqrt(loss_fn(y_pred, y_batch))
                cum_test_loss.append(loss.cpu())

            print(f'Train lost: {torch.mean(torch.from_numpy(np.array(cum_loss, dtype=np.float64)))} Val Lost: {torch.mean(torch.from_numpy(np.array(cum_test_loss, dtype=np.float64)))}')
            writer.add_scalar('data/train', torch.mean(torch.from_numpy(np.array(cum_loss, dtype=np.float64))), t)
            writer.add_scalar('data/val', torch.mean(torch.from_numpy(np.array(cum_test_loss, dtype=np.float64))), t)
            # Add loss to tracking queue
            loss_tracking.append(np.average(np.array(cum_test_loss, dtype=np.float64)))
            if len(loss_tracking) > 10:
                gradient = np.average(np.gradient(loss_tracking, axis=0))
                loss_tracking.pop(0)
                if gradient > 0:
                    print('Hit Test Loss grad limit')
                    break



        # export scalar data to JSON for external processing
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

    def create_predictions(self):
        from src.data.Analysis import JosiahAnalysis
        import pandas as pd
        import numpy as np
        # Variables to accumulate the outputs for the prediction csv
        indexes = []
        y_los = []
        y_reason = []
        print('Moving to test dataset')

        self.model.eval()
        # Run on test set:
        maxes = self.train_val_dataset.max_value_key[JosiahAnalysis.DECISION_VARIABLES]
        for i in range(len(self.test_dataset)):
            x = self.test_dataset[i]['X'].to(device=self.device)
            x_indexed = self.test_dataset[i]['I'].to(device=self.device)
            # Add missing columns as zeros
            y_pred = self.model(x)
            decoded_y_pred = np.multiply(y_pred.cpu().detach().numpy(), np.array(maxes))
            indexes.append(int(x_indexed.cpu().detach().numpy()[0]))
            y_reason.append(int(round(decoded_y_pred[0])))
            y_los.append(int(round(decoded_y_pred[1])))
            if i % 1000:
                print(f'At test i {i}')

        date = datetime.now()
        pd.DataFrame({'CASEID': indexes, 'LOS_PRED': y_los, 'REASON_PRED': y_reason}) \
            .to_csv(f'../submission/{str(date.ctime())}predictions.csv', header=False, index=False)
