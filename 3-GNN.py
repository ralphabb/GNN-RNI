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
from k_gnn import TwoMalkin, ThreeMalkin
parser = argparse.ArgumentParser()
parser.add_argument('--no-train', default=False)
parser.add_argument('-layers', type=int, default=8)
parser.add_argument('-width', type=int, default=64)
parser.add_argument('-epochs', type=int, default=500)
parser.add_argument('-dataset', type=str, default='EXP')
parser.add_argument('-learnRate', type=float, default=0.001)
args = parser.parse_args()


def print_or_log(input_data, log=True, log_file_path="logDiet.txt"):
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
        data = TwoMalkin()(data)
        data = ThreeMalkin()(data)
        data.x = F.one_hot(data.x[:, 0], num_classes=2).to(torch.float)  # Convert node labels to one-hot
        return data


# Command Line Arguments
DATASET = args.dataset
LAYERS = args.layers
EPOCHS = args.epochs
WIDTH = args.width
LEARNING_RATE = args.learnRate

MODEL = "3-GNN"   # Name of the log file
if LEARNING_RATE != 0.001:
    MODEL = MODEL + "lr" + str(LEARNING_RATE)  # Append Learning Rate if non-standard
BATCH = 20    # Batch Size
dataset = PlanarSATPairsDataset(root="Data/"+DATASET,
                                pre_transform=T.Compose([MyPreTransform()]),
                                pre_filter=MyFilter())

dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = dataset.data.iso_type_2.max().item() + 1
dataset.data.iso_type_2 = F.one_hot(
    dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
num_i_3 = dataset.data.iso_type_3.max().item() + 1
dataset.data.iso_type_3 = F.one_hot(
    dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)


class ThreeGNN(torch.nn.Module):
    def __init__(self):
        super(ThreeGNN, self).__init__()
        self.conv6 = GraphConv(num_i_3, WIDTH)
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(LAYERS):
            self.conv_layers.append(GraphConv(WIDTH, WIDTH))

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
        data.x = F.elu(self.conv6(data.iso_type_3, data.edge_index_3))
        for layer in range(LAYERS): # Number of message passing iterations we want to test over
            data.x = F.elu(self.conv_layers[layer](data.x, data.edge_index_3))
        x_3 = scatter_max(data.x, data.batch_3, dim=0)[0]

        x = x_3
        if args.no_train:
            x = x.detach()

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ThreeGNN().to(device)


def train(epoch, loader, optim):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optim.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optim.step()
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
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(   # Optional: Can Re-enable LR decay if needed
        optimizer, mode='min', factor=0.7, patience=5, min_lr=LEARNING_RATE)

    n = len(dataset) // SPLITS
    test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
    test_exp_mask = torch.zeros(len(dataset), dtype=torch.uint8)
    test_lrn_mask = torch.zeros(len(dataset), dtype=torch.uint8)

    test_mask[i * n:(i + 1) * n] = 1  # Now set the masks
    # Size must be a multiple of 4 * SPLIT
    learning_indices = [x for idx, x in enumerate(range(n * i, n * (i+1))) if x % 4 <= 1]
    test_lrn_mask[learning_indices] = 1
    exp_indices = [x for idx, x in enumerate(range(n * i, n * (i+1))) if x % 4 >= 2]
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
                     epoch+1, lr, train_loss, val_loss, test_acc, test_exp_acc, test_lrn_acc, train_acc),
                     log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
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
print_or_log('Average Across Splits', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Training Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tr_accuracies = np.mean(tr_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tr_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")

print_or_log('Testing Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tst_accuracies = np.mean(tst_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tst_accuracies[epoch]),
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
