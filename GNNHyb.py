import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from PlanarSATPairsDataset import PlanarSATPairsDataset
from k_gnn import DataLoader, GraphConv, max_pool
from k_gnn import TwoMalkin, ConnectedThreeMalkin
parser = argparse.ArgumentParser()
parser.add_argument('--no-train', default=False)
parser.add_argument('-layers', type=int, default=8)   # Number of GNN layers
parser.add_argument('-width', type=int, default=64)    # Dimensionality of GNN embeddings
parser.add_argument('-epochs', type=int, default=500)    # Number of training epochs
parser.add_argument('-dataset', type=str, default='EXP')    # Dataset being used
parser.add_argument('-randomRatio', type=float, default=1.0)   # Random ratio: 1.0 -full random, 0 - deterministic
# parser.add_argument('-clip', type=float, default=0.5)    # Gradient Clipping: Disabled
parser.add_argument('-probDist', type=str, default="n")    # Probability disttribution to initialise randomly
# n: Gaussian, xn: Xavier Gaussian, u: Uniform, xu: Xavier uniform
parser.add_argument('-normLayers', type=int, default=1)    # Normalise Layers in the GNN (default True/1)
parser.add_argument('-activation', type=str, default="tanh")    # Non-linearity used
parser.add_argument('-learnRate', type=float, default=0.001)   # Learning Rate
args = parser.parse_args()


def print_or_log(input_data, log=True, log_file_path="Debug.txt"):
    if not log:  # If not logging, we should just print
        print(input_data)
    else:  # Logging
        log_file = open(log_file_path, "a+")
        log_file.write(str(input_data) + "\r\n")
        log_file.close()  # Keep the file available throughout execution


class MyFilter(object):
    def __call__(self, data):
        return True  # No Filtering


class MyPreTransform(object):
    def __call__(self, data):
        data.x = F.one_hot(data.x[:, 0], num_classes=2).to(torch.float)  # Convert node labels to one-hot
        return data


# Command Line Arguments
DATASET = args.dataset
LAYERS = args.layers
EPOCHS = args.epochs
WIDTH = args.width
RANDOM_RATIO = args.randomRatio
DISTRIBUTION = args.probDist
ACTIVATION = F.elu if args.activation == "elu" else F.tanh
# CLIP = args.clip
LEARNING_RATE = args.learnRate

NORM = args.normLayers == 1
MODEL = "GNNHyb-"+str(args.activation)+"-"+str(RANDOM_RATIO).replace(".", ",")+"-"+str(DISTRIBUTION)+"-"+str(NORM)+"-"


if LEARNING_RATE != 0.001:
    MODEL = MODEL+"lr"+str(LEARNING_RATE)+"-"

BATCH = 20
MODULO = 4
MOD_THRESH = 1

dataset = PlanarSATPairsDataset(root="Data/"+DATASET,
                                pre_transform=T.Compose([MyPreTransform()]),
                                pre_filter=MyFilter())


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        deterministic_dims = WIDTH - int(RANDOM_RATIO * WIDTH) 
        if deterministic_dims > 0:  # Transform the deterministic dimensions using additional GraphConv
            self.conv1 = GraphConv(dataset.num_features, 32, norm=NORM)
            self.conv2 = GraphConv(32, deterministic_dims, norm=NORM)
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(LAYERS):
            self.conv_layers.append(GraphConv(WIDTH, WIDTH, norm=NORM))
        self.fc1 = torch.nn.Linear(WIDTH, WIDTH)
        self.fc2 = torch.nn.Linear(WIDTH, 32)
        self.fc3 = torch.nn.Linear(32, dataset.num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                module.reset_parameters()
            except AttributeError:
                for x in module:
                    x.reset_parameters()

    def forward(self, data):
        if int(RANDOM_RATIO * WIDTH) > 0:  # Randomness Exists
            random_dims = torch.empty(data.x.shape[0], int(RANDOM_RATIO * WIDTH))  # Random INIT
            if DISTRIBUTION == "n":
                torch.nn.init.normal_(random_dims)
            elif DISTRIBUTION == "u":
                torch.nn.init.uniform_(random_dims, a=-1.0, b=1.0)
            elif DISTRIBUTION == "xn":
                torch.nn.init.xavier_normal_(random_dims)
            elif DISTRIBUTION == "xu":
                torch.nn.init.xavier_uniform_(random_dims)
            if int(RANDOM_RATIO * WIDTH) < WIDTH:  # Not Full Randomness
                data.x1 = ACTIVATION(self.conv1(data.x, data.edge_index))
                data.x2 = ACTIVATION(self.conv2(data.x1, data.edge_index))
                data.x3 = torch.cat((data.x2, random_dims), dim=1)
            else:  # Full Randomness
                data.x3 = random_dims
        else:  # No Randomness
            data.x1 = ACTIVATION(self.conv1(data.x, data.edge_index))
            data.x3 = ACTIVATION(self.conv2(data.x1, data.edge_index))

        for layer in range(LAYERS):  # Number of message passing iterations we want to test over
            data.x3 = ACTIVATION(self.conv_layers[layer](data.x3, data.edge_index))
        x = data.x3
        x = scatter_max(x, data.batch, dim=0)[0]
        
        if args.no_train:
            x = x.detach()

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)


def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        nb_trials = 1   # Support majority vote, but single trial is default
        successful_trials = torch.zeros_like(data.y)
        for i in range(nb_trials):  # Majority Vote
            pred = model(data).max(1)[1]
            successful_trials += pred.eq(data.y)
        successful_trials = successful_trials > (nb_trials // 2)
        correct += successful_trials.sum().item()
    return correct / len(loader.dataset)

acc = []
tr_acc = []
SPLITS = 10
tr_accuracies = np.zeros((EPOCHS, SPLITS))
tst_accuracies = np.zeros((EPOCHS, SPLITS))
tst_exp_accuracies = np.zeros((EPOCHS, SPLITS))
tst_lrn_accuracies = np.zeros((EPOCHS, SPLITS))

for i in range(SPLITS):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=LEARNING_RATE)

    n = len(dataset) // SPLITS
    test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
    test_exp_mask = torch.zeros(len(dataset), dtype=torch.uint8)
    test_lrn_mask = torch.zeros(len(dataset), dtype=torch.uint8)

    test_mask[i * n:(i + 1) * n] = 1 # Now set the masks
    learning_indices = [x for idx, x in enumerate(range(n * i, n * (i+1))) if x % MODULO <= MOD_THRESH]
    test_lrn_mask[learning_indices] = 1
    exp_indices = [x for idx, x in enumerate(range(n * i, n * (i+1))) if x % MODULO > MOD_THRESH]
    test_exp_mask[exp_indices] = 1

    # Now load the datasets
    test_dataset = dataset[test_mask]
    test_exp_dataset = dataset[test_exp_mask]
    test_lrn_dataset = dataset[test_lrn_mask]
    train_dataset = dataset[1 - test_mask]

    n = len(train_dataset) // SPLITS
    val_mask = torch.zeros(len(train_dataset), dtype=torch.uint8)
    val_mask[i * n:(i + 1) * n] = 1
    val_dataset = train_dataset[val_mask]
    train_dataset = train_dataset[1 - val_mask]

    val_loader = DataLoader(val_dataset, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    test_exp_loader = DataLoader(test_exp_dataset, batch_size=BATCH) # These are the new test splits
    test_lrn_loader = DataLoader(test_lrn_dataset, batch_size=BATCH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    print_or_log('---------------- Split {} ----------------'.format(i),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
    best_val_loss, test_acc = 100, 0
    for epoch in range(EPOCHS):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        test_exp_acc = test(test_exp_loader)
        test_lrn_acc = test(test_lrn_loader)
        tr_accuracies[epoch, i] = train_acc
        tst_accuracies[epoch, i] = test_acc
        tst_exp_accuracies[epoch, i] = test_exp_acc
        tst_lrn_accuracies[epoch, i] = test_lrn_acc
        print_or_log('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
              'Val Loss: {:.7f}, Test Acc: {:.7f}, Exp Acc: {:.7f}, Lrn Acc: {:.7f}, Train Acc: {:.7f}'.format(
                  epoch+1, lr, train_loss, val_loss, test_acc, test_exp_acc, test_lrn_acc, train_acc),log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
    acc.append(test_acc)
    tr_acc.append(train_acc)

acc = torch.tensor(acc)
tr_acc = torch.tensor(tr_acc)
print_or_log('---------------- Final Result ----------------',
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()),
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Tr Mean: {:7f}, Std: {:7f}'.format(tr_acc.mean(), tr_acc.std()),
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Average Acros Splits', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Training Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tr_accuracies = np.mean(tr_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tr_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")

print_or_log('Testing Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tst_accuracies = np.mean(tst_accuracies, axis=1)
st_d_tst_accuracies = np.std(tst_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tst_accuracies[epoch])+"/"+str(st_d_tst_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")

print_or_log('Testing Exp Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tst_e_accuracies = np.mean(tst_exp_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tst_e_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")

print_or_log('Testing Lrn Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tst_l_accuracies = np.mean(tst_lrn_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tst_l_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
