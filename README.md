
## Generative Neural Networks for Classical Music Synthesis

### Authors
Christopher Fu
Connor Chong
James Bigbee

### Abstract
The goal of our project is to analyze several neural network models and use them to generate piano classical music in a style reminiscent to Bach. Each model is similar in the fact that each model reads in files, learns the classical style, and creates a complete original based on the data, but that is where the similarities end. We decide to use three different models: Recurrent Neural Networks (RNNs), Dilated Convolutional Neural Networks (DCNNs), and Generative Adversarial Networks (GANs). \\
We hoped that these unique models can each capture particular aspects of classical music: namely rhythm, chord harmonization, and chord progression. Although these traits are present in all music, classical music has strong structural characteristics that makes it unique as a music genre. We were able to achieve greater success with the RNNs and the GAN models, and you can listen to the samples that we produce in the paper. We can contrast the furthest reaches of state-of-the-art computational creativity with humanity’s classical expression, and discuss further improvements to our models going forwards.

### Implementation
Our procedure can be broken down into four main subsections: preprocessing, feature-mapping and accurate representations, ML architecture design, and then evaluation of the results of music generation. We will talk about our thought process behind our decisions made at each of these steps, talk about design trade-offs, and explore avenues of improvement. 

We gathered a dataset of over 500 MIDI files related to Bach's compositions (in the midi_files folder), and have already pre-processed them, transposing each composition into the same key and filtering out compositions that did not have the same time signature (we chose the time signatures 3/4 and 4/4 for regularity).

## The Models

### RNN (Recurrent Neural Network)
The fist model we used to attempt to generate music was a Recurrent neural net or RNN. Unlike a standard neural network a RNN output not only a prediction but also a history vector, which is used on the next prediction and history output. The addition of history allows for prediction to be made using not only the current data but also the past. Therefore RNN are especially well suited to tasks that rely on temporal data such as music. We tested two types of RNN: GRU and LSTM, both of which attempt to solve a problem of vanilla RNN have of not being able to carry important data the past through do to vanishing gradient.
For both GRU and LST we implemented them as character RNN, where a string of characters is fed in and the next character is predicted. To do this after the data has been transposed it is converted into a single string with spaces marking the end of each 16th note and words representing different notes being played at once with a characters representing each note. These strings are fed into our recurrent neural net. Once the RNN is trained we first select a character and then generate an output distribution based on the current character and history and choose the next character based on that and continue this process for some number of characters. Then the string of characters is converted back into a midi file.
We first Tested the GRU. We trained over 15K epochs taking random sample of 100 characters of the training data. 
We found that loss plateaued after about the 800th epoch. To test how well we preformed we also measured the perplexity of random samples of both training and validation to make sure we were not over fitting. As seen the perplexity for both is about the same meaning the model views the creation of either in general as equally likely.

The Results of the music generation where good though not what one would call good music. Though the generated music had short term progression and in general notes that worked well together in the long term there was very little progression, and there were occasional moments of dissonance. Also due to the nature of the conversion the usual most likely note to follow a note was itself and the most likely character representing rest. Therefore occasionally very sparse pieces were occasionally created. For better or for worse this model also sounded the most like the style Bach. Below are an two examples one of each models generation of music: 

<a href ="https://s3.amazonaws.com/website-connor-chong/15kTemp.8GRU.mid">GRU generated</a>

<a href="https://s3.amazonaws.com/website-connor-chong/LSTMMUSIC2.midi">LSTM generated</a>



### Dilated Convolutional Neural Network
The second model we implemented was a modified convolutional neural network. This new technique utilizes a model usually used for 2D visual imagery and applies it in a clever way to handle audio. The first major issue with using raw audio with traditional convolutional neural networks is the high-dimension nature of the wav files. Typically, each second of audio is 16kHz (meaning 16000 samples per second, which is a ridiculously huge amount of generation needed) where each datapoint is highly dependent on previous data points. If we are to use traditional convolutional filters, we would need an absurd number of layers to obtain a receptive field capable of capturing the necessary features—this is clearly impractical. Rather than create a huge tower of filters that would skyrocket training times, we create a receptive field by making the filters very “wide.” By using dilations, we can create an exponentially-expanding receptive field every layer we go down; by doing this, we can keep a reasonably small number of convolutional layers without sacrificing the receptive field we need to capture the sequential nature of music.

In addition to this dilation, this technique also implements gated activation functions similarly to RNNs like LSTM. In our implementation, we use a logistic sigmoid and a tanh function. The reasoning behind using these gates are to create learning over time, making future training affected by all the information the model has received before as a guideline. To help out with convergence, we also implement residual and skip-connections.

On top of all the above, researchers for this model have targeted three optimizations that improve general runtime of generation. The first optimization tackles the recalculation cases, using caches to reduce an exponential computational complexity to a nearly linear complexity. In the naïve approach, every sample recalculates all the dilations. Instead, We cache previous calculations for quick retrieval, reducing the number of total calculations we need to make by an exponential factor.

However, in practice, due to the huge amount of generation necessary (as stated before, 16000 samples for just one second) is too much for a typical system to handle. One solution is to use inverse-autoregressive flows to make all values computable in paralle. The Inverse-autoregressive flows allows us to augment our Gaussian model with invertible mappings (for ease of calculation, invertible mappings with easy Jacobian calculations) to create a multi-modal probability distribution that can better capture our model predictions. Another improvement is to use a modified adversarial networks where a student models after a teacher, but more efficiently.

Due to the incredible complexity of the algorithms we are running, I was difficult to get much testing done--it took an extremely long time to get a training set in, even utilizing the full GPU power with NVIDIA’s CUDA. We originally planned to try out the cache version as well as the adversarial version and compare the two; however, training our first version of the cache method itself took over four days for one instance, without generating more than roughly a second of audio. By the time we had a working model, there wasn’t enough time to train a adversarial model as well as generate audio. Unfortunately, due to the time constraints of this final project, we were unable to look into the adversarial model very much.

As a result, we implemented the fast version utilizing cache queues. We used the same gates, skip-layers, and dilation tactics in the original version. Our model sports five dilated convolutional layers with 32 dilation channels, 32 residual channels, 1024 skip channels, and a bias. Our hyperparameters were a learning rate of 0.001, ADAM optimizer, and cross-entropy loss function. Our final model ran for one day, 11 hours, 18 minutes and 11 seconds, going through over 187 thousand steps.

Even though this was utilizing the fast cache queue version, generating over 16,000 samples per second was impossible for the laptops/desktops we had access to. We only managed to generate two audio files, but each had a length of half a second. On the other hand, through tensorboard, we have access to all the weights, biases, and loss calculations for every step.
Our loss decreased steadily from 2.962 to roughly 2.7, and our validation loss decresed from 3.145 to 2.938. Although this is a slight improvement, it is clear that the algorithm was far from running to completion, even after nearly two days of straight operation. It is also clear from the graph that the gradient of the loss was still quite high, so theoretically our model could have performed significantly better if given the time and the computational power.

In conclusion, this implementation of dilated convolutional neural networks works well in theory, but is quite cumbersome in terms of training. Although we were unable to implement the best version, it still took a ridiculous amount of time to train and generate audio samples. We were limited in the scope of computational power and time, but the end result from this model was still a bit lower than expected. Although the model did do its best to learn Bach, it seems like faster implementations, coupled with stronger GPUs, and simply more time is necessary to bring this algorithm to its highest potential.

### GAN (Generative Adversial Network)
A breakthrough development in NN performance has been through the use of Generative Adversarial NNs. We decided to take advantage of the powerful generative ability of the GAN to apply its usual application, image synthesis, towards music synthesis. 

#### How it works
The GAN is made of two adversarial neural networks. The GAN is a minimax tug-of-war in which the neural network generator seeks to minimize the gradient (through gradient descent) while the neural network discriminator seeks to maximize it.


#### Implementation
I modeled my GAN off existing models, taken from the original <a href="https://arxiv.org/abs/1406.2661">paper here </a>, with the corresponding <a href="https://github.com/goodfeli/adversarial">code here</a>. I modified each of the steps, however, based off <a href="htts://github.com/soumith/ganhacks">several recommendations here </a> that explain several improvements to help GAN stability. 

The Generator:
The generator is meant to be more powerful than the discriminator, and thus I added more connected layers with leaky ReLu activation functions. Similar to the paper, I decided to use four connected hidden layers of size 128, 256, 512, and 1028 respectively, adding Batch Normalization and Leaky ReLu activation functions. Then I added a Tanh activation function at the output, as talked about in the recommended improvements page. 

The Discriminator:
The discriminator is a smaller neural network with 2 connected hidden layers of size 512 and 256 respectively. I then added a Sigmoid activation function at the output, to help produce 1 or 0 labels which would tell if the input is real or generated data. 

We added a balancer to control the loss functions of the generator and the discriminator. If the loss of the generator was too high during an epoch, we would run additional generator training steps and backpropogate the loss to make the loss of the generator lower, and likewise for the discriminator. Thus, this would help control and prevent GAN model collapse, a common phenomena in GAN models when the generator and discriminator diverge indefinitely. 

#### Results
The GAN model is insanely sensitive to slight differences in the tuning parameters. Running 5 simultaneous Colab instances with GPU, I was able to sample around 70 different models of 1000+ epochs each with slightly different learning rates and other tuning parameters. I mainly modified the generator learning rate and the discriminator learning rate, to control how strong each learner was compared to the other. Still, even with my added balancer and careful tuning, probably 80\% of the models collapsed after a few hundred epochs. Here is an example:


<a href="https://s3.amazonaws.com/website-connor-chong/gan_epoch1480c1(2).mid">Example of a collapsed GAN</a>
We can clearly see that in the above diagram, the generator and discriminant loss fluctuate rapidly after epoch 300. However, I had several successes in my models as well. 

In this model, with the generator learning rate at .0008 and the discriminator learning rate at .00001 (so the generator learns roughly 80x faster), I was able to achieve a decently evenly matched discriminator and generator. Other than the blip in the middle of the graph, the generator and discriminator converge to roughly equal loss values. You can listen to the progression of the midi from 500 epochs to 1000 epochs. \\
<a href="https://s3.amazonaws.com/website-connor-chong/gan_epoch500c1lr000800001.mid.mid">500 Epochs of Training</a>,   
<a href="https://s3.amazonaws.com/website-connor-chong/gan_epoch980c1lr000800001.mid.mid">1000 Epochs of Training</a>

The GAN models are able to maintain chord progression and harmony. Here is another example of a successful model with the tuning set at .0005 and .00001 for the generator and the discriminator, respectively. 
<a href="https://s3.amazonaws.com/website-connor-chong/gan_epoch980c1+.0005+.00001.mid">1000 Epochs of Training</a>
In this midi, there exists a variation on a theme, where the piece replays a portion of the song, but in a slightly different way. Bach is very well known for his variations on themes in many of his pieces, further showing the GAN's ability to generate many musical features present in classical music. 



