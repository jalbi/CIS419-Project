import argparse
import os
import numpy as np
import math
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as utils
from midi_to_matrix import midiToNoteStateMatrix, noteStateMatrixToMidi

# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--hidden_size", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# opt = parser.parse_args()
# print(opt)

midi_directory = './midi_files'
num_epochs = 3000
batch_size = 32
learning_rate_gen = .002
learning_rate_dis = .00005
hidden_size = 100
sample_interval = 20
b1 = .5
b2 = .9999
midi_len = 250
features = 156
channels = 1

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(hidden_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 250*156),
            nn.Tanh()
        )

    def forward(self, z):
        mat = self.model(z)
        mat = mat.view(mat.size(0), 250, 156)
        # print("forward G", mat.shape)
        return mat


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(features*midi_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, mat):
        mat_flat = mat.view(mat.size(0), -1)
        validity = self.model(mat_flat)
        # print("forward D", validity.shape)
        return validity


def midiToMatrix():
    midiFiles = []
    for filename in os.listdir(midi_directory):
        if filename.lower().endswith(".mid"):
            matrix = np.array(midiToNoteStateMatrix(os.path.join(midi_directory, filename)))
            x, y, z = matrix.shape
            if(x >= 250):
                matrix = matrix[:midi_len]
                matrix = matrix.reshape(-1, 2*78)
                # print(matrix.shape)
                midiFiles.append(matrix)
    return midiFiles


def dataLoader():
    m = midiToMatrix()
    dataset = Dataset(np.array(m)) 
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return loader

############################################################
# Extracting and loading data
############################################################
class Dataset(Dataset):
    def __init__(self, X):
        self.len = len(X)           
        if torch.cuda.is_available():
            self.x_data = torch.from_numpy(X).float().cuda()
        else:
            self.x_data = torch.from_numpy(X).float()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx]

def backToMidi(matrices, epoch):
    counter = 0
    x, y, z = matrices.shape
    for i in range(0, x, 1):
        newArr = np.array(matrices[i])
        # print(newArr.tolist())
        newArr = np.round(newArr)
        newArr = newArr.reshape((250, 78, 2))
        name = "gan_epoch" + str(epoch) + "c" + str(counter)
        noteStateMatrixToMidi(newArr, name)
        counter +=1

if __name__ == "__main__":
    data = dataLoader()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate_gen, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_dis, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    discriminantLoss = []
    generatorLoss = []

    for epoch in range(num_epochs):
        batchGenLoss = []
        batchDisLoss = []
        for i, mat in enumerate(data):
            # print("matrix shape: ", mat.shape)
            # Adversarial ground truths
            valid = Variable(Tensor(mat.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(mat.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_mat = Variable(mat.type(Tensor))
            #  Train Generator
            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], hidden_size))))
            # Generate a batch of matricies
            gen_mat = generator(z)
            # print("gen shape", gen_mat.shape)
            # Loss measures generator's ability to fool the discriminator
            adversarial_loss = torch.nn.BCELoss()
            g_loss = adversarial_loss(discriminator(gen_mat), valid)
            g_loss.backward()
            optimizer_G.step()


            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_mat), valid)
            fake_loss = adversarial_loss(discriminator(gen_mat.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            
            batchGenLoss.append(d_loss.item())
            batchDisLoss.append(g_loss.item())
            if epoch % sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(data), d_loss.item(), g_loss.item())
                )
                # print(gen_mat.detach().numpy().shape)
                backToMidi(gen_mat.detach().numpy()[:3], epoch)
            while(g_loss.item() > .8):
                valid = Variable(Tensor(mat.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(mat.shape[0], 1).fill_(0.0), requires_grad=False)
                # Configure input
                real_mat = Variable(mat.type(Tensor))
                #  Train Generator
                optimizer_G.zero_grad()
                z = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], hidden_size))))
                # Generate a batch of matricies
                gen_mat = generator(z)
                # print("gen shape", gen_mat.shape)
                # Loss measures generator's ability to fool the discriminator
                adversarial_loss = torch.nn.BCELoss()
                g_loss = adversarial_loss(discriminator(gen_mat), valid)
                g_loss.backward()
                optimizer_G.step()

            while(d_loss.item() > .8):
                optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_mat), valid)
                fake_loss = adversarial_loss(discriminator(gen_mat.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

        discriminantLoss.append(sum(batchDisLoss)/len(batchDisLoss))
        generatorLoss.append(sum(batchGenLoss)/len(batchGenLoss))
    
    epoch1 = []
    for i in range(0, num_epochs, 1):
        epoch1.append(i)
    
    df=pd.DataFrame({'epoch': epoch1, 'Generator Loss': generatorLoss, 'Discriminant Loss': discriminantLoss})
    plt.plot( 'epoch', 'Generator Loss', data=df, marker='', color='green', linewidth=2)
    plt.plot( 'epoch', 'Discriminant Loss', data=df, marker='', color='red', linewidth=2)
    plt.legend()
    plt.show()
