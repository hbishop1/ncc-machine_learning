import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

dset_train = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]))

dset_test = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]))

dset = torch.utils.data.ConcatDataset([dset_test,dset_train])

indicies=[]
for i in range(len(dset)):
    if dset[i][1] in [2,7] and i%50 == 0:
        indicies.append(i)

train_dataset = torch.utils.data.Subset(dset,indicies)

train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')


class VAE(nn.Module):
    def __init__(self, intermediate_size=256, hidden_size=5):
        super(VAE, self).__init__()


        # encoder
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.batch2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.batch4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4 * 4 * 128, intermediate_size)
        self.batch5 = nn.BatchNorm1d(intermediate_size)

        # latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

        # decoder
        self.fc3 = nn.Linear(hidden_size, intermediate_size)
        self.batch6 = nn.BatchNorm1d(intermediate_size)

        self.fc4 = nn.Linear(intermediate_size, 16 * 16 * 128)
        self.batch7 = nn.BatchNorm1d(16*16*128)

        self.deconv1 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batch8 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch9 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.batch10 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        out = self.batch1(F.relu(self.conv1(x)))
        out = self.batch2(F.relu(self.conv2(out)))
        out = self.batch3(F.relu(self.conv3(out)))
        out = self.batch4(F.relu(self.conv4(out)))
        out = out.view(out.size(0), -1)
        h1  = self.batch5(F.relu(self.fc1(out)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3  = self.batch6(F.relu(self.fc3(z)))
        out = self.batch7(F.relu(self.fc4(h3)))
        out = out.view(out.size(0), 128, 16, 16)
        out = self.batch8(F.relu(self.deconv1(out)))
        out = self.batch9(F.relu(self.deconv2(out)))
        out = self.batch10(F.relu(self.deconv3(out)))
        out = torch.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


N = VAE().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.001)
num_epochs = 50

# VAE loss has a reconstruct16ion term and a KL divergence term summed over all elements and the batch
def vae_loss(p, x, mu, logvar):
    BCE = F.binary_cross_entropy(p.view(-1, 32 * 32 * 3), x.view(-1, 32 * 32 * 3), reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

open('results_pegasus.txt','w').close()

# training loop, feel free to also train on the test dataset if you like
for epoch in range(1,num_epochs+1):

    print('-' * 10)
    print('Epoch {}/{}'.format(epoch,num_epochs))
    
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)
    test_loss_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(1000):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        optimiser.zero_grad()
        
        p, mu, logvar = N(x)
        loss = vae_loss(p, x, mu, logvar)
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)

    print('Train Loss: {:.4f}'.format(train_loss_arr.mean()))

    with open('results_pegasus.txt','a') as results:
        results.write('Epoch {}/{} \n'.format(epoch,num_epochs))
        results.write('Train Loss: {:.4f} \n'.format(train_loss_arr.mean()))

    epoch = epoch+1

with torch.no_grad():
    sample = torch.randn(64, 5).to(device)
    sample = N.decode(sample).cpu()
    save_image(sample.view(64, 3, 32, 32),'pegasus.png')

