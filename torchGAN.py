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

#training parameters
midi_directory = './midi_files'     # Directory of midi file dump (~500 samples)
num_epochs = 1000                   # Number of training epochs
batch_size = 64                     # Batch size, 64 is more accurate but 128 is faster
learning_rate_gen = .002            # Learning rate of the generator, one of the primary tuners of the model
learning_rate_dis = .00005          # Learning rate of the discriminator, of of the primary tuners of the model
hidden_size = 100                   # Size of the latent dimension space (how many midi files we generate at once to feed to the discriminator)
sample_interval = 20                # Sample interval to save the current midi generated files
b1 = .5                             # Parameters of the Adam optimizer
b2 = .9999                          # Parameters of the Adam optimizer
midi_len = 250                      # Extract first 250 features of the midi file
features = 156                      # Features in the midi matrix
channels = 1                        # Number of channels

# Activate CUDA
cuda = True if torch.cuda.is_available() else False

#initialization function
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

        # define a neural network with 3 hidden layers of 128, 256, 512, and 1024 respectively
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
            nn.Tanh() #Tanh to improve performance
        )

    def forward(self, z):
        mat = self.model(z)
        mat = mat.view(mat.size(0), 250, 156) #reshape
        return mat


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Traditional neural network with 2 hidden layers of size 512 and 256, respectively
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
        return validity

# convert a midi file to the feature matrix using the midiToNoteStateMatrix function
def midiToMatrix():
    midiFiles = []
    for filename in os.listdir(midi_directory):
        if filename.lower().endswith(".mid"):
            matrix = np.array(midiToNoteStateMatrix(os.path.join(midi_directory, filename)))
            x, y, z = matrix.shape
            if(x >= 250):
                matrix = matrix[:midi_len]
                matrix = matrix.reshape(-1, 2*78)
                midiFiles.append(matrix)
    return midiFiles

# load the data through a custom Dataset class
def dataLoader():
    m = midiToMatrix() 
    dataset = Dataset(np.array(m)) 
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return loader

# Pulled from the Custom Class defined in HW4 of this class to load the matrices
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

# convert back to midi file
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
    # Load data
    data = dataLoader()
    # Init Generator and Discriminator
    generator = Generator()
    discriminator = Discriminator()

    
    if cuda:
        generator.cuda()
        discriminator.cuda()
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
            # Generate valid and fake data tensors
            valid = Variable(Tensor(mat.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(mat.shape[0], 1).fill_(0.0), requires_grad=False)
            real_mat = Variable(mat.type(Tensor))
            


            #  Train Generator
            optimizer_G.zero_grad()

            # Create a noise vector drawn from a Gaussian distribution to feed into generator
            z = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], hidden_size))))

            # Generate midi matrices to feed into discriminator
            gen_mat = generator(z)
            
            # Loss which measures how well the discriminator performs
            adversarial_loss = torch.nn.BCELoss()
            if cuda: # activate cuda
                adversarial_loss.cuda()
            g_loss = adversarial_loss(discriminator(gen_mat), valid)

            # Backpropogate
            g_loss.backward()
            optimizer_G.step()



            # Train Discriminator
            optimizer_D.zero_grad()
            # Compare loss from real data and loss from generator's output
            real_loss = adversarial_loss(discriminator(real_mat), valid)
            fake_loss = adversarial_loss(discriminator(gen_mat.detach()), fake)
            # Take the average loss of the real and fake loss
            d_loss = (real_loss + fake_loss) / 2

            # Backpropogate
            d_loss.backward()
            optimizer_D.step()
            
            # If discriminator loss is too high, train the discriminator more
            while(d_loss.item() > .8):
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_mat), valid)
                fake_loss = adversarial_loss(discriminator(gen_mat.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

            # If generator loss is too high, train the generator more
            while(g_loss.item() > .8):
                # see above, basically runs generator steps again
                valid = Variable(Tensor(mat.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(mat.shape[0], 1).fill_(0.0), requires_grad=False)
                real_mat = Variable(mat.type(Tensor))
                optimizer_G.zero_grad()
                z = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], hidden_size))))
                gen_mat = generator(z)
                g_loss = adversarial_loss(discriminator(gen_mat), valid)
                g_loss.backward()
                optimizer_G.step()

            batchGenLoss.append(d_loss.item())
            batchDisLoss.append(g_loss.item())

            # Every sample interval, print out current D loss and G loss and save midis at this point
            if epoch % sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(data), d_loss.item(), g_loss.item())
                )
                backToMidi(gen_mat.detach().numpy()[:3], epoch) # create three midi files 
    
        discriminantLoss.append(sum(batchDisLoss)/len(batchDisLoss))
        generatorLoss.append(sum(batchGenLoss)/len(batchGenLoss))
    
    # plot the graph of generator and discriminator loss
    epoch1 = []
    for i in range(0, num_epochs, 1):
        epoch1.append(i)
    df=pd.DataFrame({'epoch': epoch1, 'Generator Loss': generatorLoss, 'Discriminant Loss': discriminantLoss})
    plt.plot( 'epoch', 'Generator Loss', data=df, marker='', color='green', linewidth=2)
    plt.plot( 'epoch', 'Discriminant Loss', data=df, marker='', color='red', linewidth=2)
    plt.legend()
    plt.show()
