import torch.optim as optim
import torch.utils.data
import time
import torch.nn.functional as torch_fun
from FinalRunfiles.logging import Custom_Logger
from FinalRunfiles.modules import *

class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 logger=Custom_Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 use_cuda=False):

        self.model = model
        self.dataset = dataset
        self.lr = lr

        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.logger = logger
        self.logger.trainer = self

        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval

        if use_cuda:
            self.dtype = torch.cuda.FloatTensor
            self.ltype = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor # data type
            self.ltype = torch.LongTensor # label type

        self.dataloader = None

    def train(self,
              batch_size=32,
              epochs=10,
              curr_step=0):
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      pin_memory=False)

        for curr_epoch in range(epochs):
            print("epoch", curr_epoch)
            for (i, target) in iter(self.dataloader):
                print(curr_step)
                target = Variable(target.view(-1).type(self.ltype))
                output = self.model(Variable(i.type(self.dtype)))

                loss = torch_fun.cross_entropy(output.squeeze(), target.squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.item()

                self.optimizer.step()
                curr_step += 1

                if curr_step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string)
                self.logger.log(curr_step, loss)

    def validate(self):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        for (i, target) in iter(self.dataloader):
            i = Variable(i.type(self.dtype))
            target = Variable(target.view(-1).type(self.ltype))

            output = self.model(i)
            loss = torch_fun.cross_entropy(output.squeeze(), target.squeeze())
            total_loss += loss.data[0]

            accurate_classifications += torch.sum(torch.eq(target, torch.max(output, 1)[1].view(-1))).data[0]
        avg_loss = total_loss / len(self.dataloader)
        avg_acc = accurate_classifications / (len(self.dataset)*self.dataset.target_length)
        self.dataset.train = True
        self.model.train()
        return avg_loss, avg_acc


def generate_audio(model,
                   length=8000,
                   temperatures=[0., 1.]):
    samples = []
    for temp in temperatures:
        samples.append(model.generate_fast(length, temperature=temp))
    samples = np.stack(samples, axis=0)
    return samples
