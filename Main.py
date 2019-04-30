"""
    Run this file to create the final product
"""

from FinalRunfiles.model import *
from FinalRunfiles.audio_data import Dataset
from FinalRunfiles.training import *
from FinalRunfiles.logging import *
import IPython.display as ipd

# initialize cuda option
use_cuda = torch.cuda.is_available()


model = Model(layers=5,
              blocks=3,
              dilation_channels=32,
              residual_channels=32,
              skip_channels=1024,
              end_channels=512,
              output_length=16,
              dtype=torch.cuda.FloatTensor,
              bias=True)
model.cuda()

#model = load_latest_model_from('snapshots', use_cuda=use_cuda)

data = Dataset(filename='Runfiles/data/dataset.npz',
               length=model.receptive_field + model.output_length - 1,
               t_length=model.output_length,
               dir='train_samples/bach_chaconne',
               stride=500)


def gen(step):
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("___START GENERATING___")
    samples = generate_audio(gen_model,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("___GENERATION COMPLETE!___")
logger = TbLog(interval=200,
               validation=400,
               gen=1000,
               gen_f=gen,
               dir="logs/chaconne_model")


newpath = 'C:/Users/diego/Desktop/CHRIS FU SHIT/Example/snapshots'
trainer = Trainer(model=model,
                  dataset=data,
                  lr=0.001,
                  dir=newpath,
                  filename='chaconne_model',
                  snapshot=1000,
                  logger=logger,
                  use_cuda=use_cuda)

print('start training...')
trainer.train(batch_size=16,
              epochs=5)


start_data = data[250000][0] # use start data from the data set
start_data = torch.max(start_data, 0)[1] # convert one hot vectors to integers


def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")
generated = model.generate_fast(num_samples=160000,
                                 first_samples=start_data,
                                 progress_callback=prog_callback,
                                 progress_interval=1000,
                                 temperature=1.0,
                                 regularize=0.)


ipd.Audio(generated, rate=16000)