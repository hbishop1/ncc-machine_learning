import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',]

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomResizedCrop(size=32,scale=(0.5,1)),
        torchvision.transforms.ToTensor()
    ])),

shuffle=True, batch_size=16, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
shuffle=False, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print('> Size of training dataset: ', len(train_loader.dataset))
print('> Size of test dataset: ', len(test_loader.dataset))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_loader.dataset[i][0].permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_loader.dataset[i][1]])

plt.show()

# define the model (a simple classifier)

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class MyNetwork(nn.Module):
    def __init__(self):

        super(MyNetwork, self).__init__()
        layers = nn.ModuleList()

        layers.append(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(128))

        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(256))

        layers.append(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(384))

        layers.append(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(384))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layers.append(nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(512))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(512))

        layers.append(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(1024))    

        layers.append(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layers.append(Flatten())

        layers.append(nn.Linear(in_features=1024*8*8, out_features=2048))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(2048))

        layers.append(nn.Linear(in_features=2048, out_features=100))

        self.layers = layers

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

N = MyNetwork().to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

N.apply(weights_init)

print('> Number of network parameters: ', len(torch.nn.utils.parameters_to_vector(N.parameters())))

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.00005, weight_decay=0.005)
num_epochs = 500
logs = {'train_acc':[],'train_loss':[],'test_acc':[],'test_loss':[]}
#liveplot = PlotLosses()

open('results1.txt','w').close()


for epoch in range(1,num_epochs+1):

    print('-' * 10)
    print('Epoch {}/{}'.format(epoch,num_epochs))
    
    # arrays for metrics
    train_loss_arr = np.zeros(0)
    train_acc_arr = np.zeros(0)
    test_loss_arr = np.zeros(0)
    test_acc_arr = np.zeros(0)

    N.train()

    # iterate over some of the train dateset
    for i in range(1000):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        optimiser.zero_grad()
        p = N(x)
        pred = p.argmax(dim=1, keepdim=True)
        loss = torch.nn.functional.cross_entropy(p, t)
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)
        train_acc_arr = np.append(train_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    N.eval()

    # iterate entire test dataset
    for x,t in test_loader:
        x,t = x.to(device), t.to(device)

        p = N(x)
        loss = torch.nn.functional.cross_entropy(p, t)
        pred = p.argmax(dim=1, keepdim=True)

        test_loss_arr = np.append(test_loss_arr, loss.cpu().data)
        test_acc_arr = np.append(test_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    with open('results1.txt','a') as results:
        results.write('Epoch {}/{} \n'.format(epoch,num_epochs))
        results.write('Train Loss: {:.4f} Train Acc: {:.4f} \n'.format(train_loss_arr.mean(),train_acc_arr.mean()))
        results.write('Test Loss: {:.4f} Test Acc: {:.4f} \n'.format(test_loss_arr.mean(),test_acc_arr.mean()))


    print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(train_loss_arr.mean(),train_acc_arr.mean()))
    print('Test Loss: {:.4f} Test Acc: {:.4f}'.format(test_loss_arr.mean(),test_acc_arr.mean()))

    logs['train_acc'].append(train_acc_arr.mean())
    logs['train_loss'].append(train_loss_arr.mean())
    logs['test_acc'].append(test_acc_arr.mean())
    logs['test_loss'].append(test_loss_arr.mean())

    with open('logs.p', 'wb') as fp:
        pickle.dump(logs, fp)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = '#335599' if predicted_label == true_label else '#ee4433'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('#ee4433')
    thisplot[true_label].set_color('#335599')

test_images, test_labels = next(test_iterator)
test_images, test_labels = test_images.to(device), test_labels.to(device)
test_preds = torch.softmax(N(test_images).view(test_images.size(0), len(class_names)), dim=1).data.squeeze().cpu().numpy()

num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, test_preds, test_labels.cpu(), test_images.cpu().squeeze().permute(1,3,2,0).contiguous().permute(3,2,1,0))
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, test_preds, test_labels)