import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import unidecode
import string
import random
import os
import statistics
import matplotlib.pyplot as plt

file = unidecode.unidecode(open('train.txt').read())
v_file = unidecode.unidecode(open('val.txt').read())
vl = len(vf)
file_len = len(file)
all_chars = string.printable


def random_chunk(chunk_len = 200):
    start = random.randint(0, file_len - chunk_len)
    return file[start:start + chunk_len + 1]

def random_chunk_v(chunk_len = 200):
    start = random.randint(0, vl - chunk_len)
    return v_file[start:start + chunk_len + 1]

def rst(chunk_len = 200):    
    chunk = random_chunk(chunk_len)
    target = char_tensor(chunk[1:])
    inp = char_tensor(chunk[:-1])
    return inp, target

def char_tensor(string):
    tensor = torch.empty(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_chars.index(string[c])
    return Variable(tensor)

def avg_plex():
    pa = 0
    pv = 0
    t = []
    v = []
    for i in range(100):
        t.append(perplexity(random_chunk()[:40]))
        v.append(perplexity(random_chunk_v()[:40]))
    return(statistics.median(t),statistics.median(v))

def gen(model, primer = '\n', length = 100, temp = .8):
    #intitalize hidden layer and make start a tensor 
    hid = model.init_hidden()
    pinp = char_tensor(primer)
    pred = primer
    for c in range(len(primer) - 1):
        hid = decoder(pinp[c],hid)[1]
    inp = pinp[-1]
    for c in range(length):
        output, hid = model(inp, hid)
        pred_char = string.printable[torch.multinomial(output.data.view(-1).div(temp).exp(), 1)[0]]
        pred += pred_char
        inp = char_tensor(pred_char)
    return pred

def perplexity(st, hidden_len=100, temperature=0.8): 
    inp = char_tensor(st)
    total = 1     
    for i in range (len(st) -1):
        if (i % hidden_len == 0):
            hid = decoder.init_hidden()
        out, hid = decoder(inp[i], hid)
        out_dist = out.data.view(-1).div(temperature).exp()
        probability = (out_dist[inp[i + 1]].item())
        total /= probability
    return math.pow(total, 1/len(st))
        
class Char_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, model = 'gru'):
        super(Char_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.model = model
        self.encoder = nn.Embedding(input_size, hidden_size)
        if (self.model == 'gru'):
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif (self.model == 'lstm'):
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax=nn.LogSoftmax()
    
    def forward(self, inp, hidden):
        inp = self.encoder(inp.view(1, -1))
        output, hidden = self.rnn(inp.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / chunk_len


chunk_len = 200
n_epochs = 10000
print_epoch = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.0005

decoder = Char_RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
all_losses = []
pv = 0
pt = 0
per_train = []
per_val = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = train(*rst(chunk_len))       
    loss_avg += loss

    if epoch % print_epoch == 0:
        print('[(%d %d%%) %.4f]' % (epoch, epoch / n_epochs * 100, loss))
        print(gen(decoder, '\n', 200)[1:])
        pt, pv = avg_plex()
        print("perplexity train:", pt, "perplexity val:", pv)
        print()
        all_losses.append(loss_avg / plot_every)
        per_val.append(pv)
        per_train.append(pt)
        loss_avg = 0

get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.plot(all_losses)
plt.figure()
plt.plot(per_val[10:], 'b-', label = 'val')
plt.plot(per_train[10:], 'y-', label ='train')


decoder.save(torch.save(os.getcwd()+"/music_gen"))


#decoder = Char_RNN(n_characters, hidden_size, n_characters, n_layers)
#decoder.load_state_dict(torch.load(os.getcwd()+"/music_gen"))
