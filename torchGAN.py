import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as utils
from midi_to_matrix import midiToNoteStateMatrix, noteStateMatrixToMidi
# os.makedirs("images", exist_ok=True)

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# opt = parser.parse_args()
# print(opt)

midi_directory = './midi_files'
num_epochs = 200
batch_size = 32
learning_rate = .0002
matrix_length = 10000
hidden_size = 100
sample_interval = 20
b1 = .5
b2 = .995
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

        self.init_size = features // 4
        self.l1 = nn.Sequential(nn.Linear(hidden_size, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        mat = self.conv_blocks(out)
        return mat


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = features // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, mat):
        out = self.model(mat)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def adversarial_loss():
    return torch.nn.BCELoss()

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
        shuffle=True,
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
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for epoch in range(num_epochs):
        for i, mat in enumerate(data):
            # Adversarial ground truths
            valid = Variable(Tensor(mat.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(mat.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_mat = Variable(mat.type(Tensor))

            #  Train Generator
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (mat.shape[0], latent_dim))))

            # Generate a batch of matricies
            gen_mat = generator(z)

            # Loss measures generator's ability to fool the discriminator
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
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            if epoch % sample_interval == 0:
                backToMidi(gen_mat.data[:5], epoch)

def backToMidi(matrices, epoch):
    counter = 0
    for m in matrices:
        m = m.reshape(-1, 78, 2)
        name = "gan_epoch" + str(epoch) + "c" + str(counter)
        noteStateMatrixToMidi(m, name)
        counter +=1