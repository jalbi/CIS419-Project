
## CIS 419/519 Final Project: Music Generation with NNs

### Authors
Christopher Fu
Connor Chong
James Bigbee

### Abstract
The goal of our project is to create a neural network that can produce piano music similar to Bach. This is slightly different from text classification--we will be using the algorithm to read in files, learn the classical style, and create a complete original based on the data. We are using neural networks as the base of our algorithm, specifically LSTM and CNNs. 
We are hoping that the neural net can capture three key aspects of classical music: rhythm, chord harmonization and progression, and phrasing. Although these traits are present in all music, classical music has a very specific take on these characteristics that makes it unique as a music genre. From there, we want the algorithm to use a generalized (neither over-fitted or under-fitted) sense of these three attributes to generate unique songs. While we do not expect it to be very good, we hope to see hints of learning. We wish to explore the furthest reaches of computational creativity, seeing if it can hold a candle to humanity’s classical expression.

### Implementation
Our procedure can be broken down into four main subsections: preprocessing, feature-mapping and accurate representations, ML architecture design, and then evaluation of the results of music generation. We will talk about our thought process behind our decisions made at each of these steps, talk about design trade-offs, and explore avenues of improvement. 

Our first step is data collection, where we are using MIDI files, which is a standardized communication protocol for electronic musical notation. Our pipeline requires adequate number of samples, and so for the purpose of this project we have decided to use Bach keyboard pieces for our training purposes, to have some sort of regularity in style (specifically Baroque composing style), but to have some sort of variation between pieces.  
Our next step is processing these files into feature vectors we can actually use in our LSTM and CNN. As we have seen in class, a CNN takes in a 2D matrix--which is usually supposed to represent an image--and performs many layers of convolution, regularization, and max pooling before the fully-connected layers. Current research literature in music generation have used a matrix where the X axis is time and the Y axis is hertz (i.e. middle A would show up as 440Hz). We decided to take a different approach, using the notes instead of pitch (A4 instead of 440Hz) and quantization (modeling every sixteenth note as an interval, rather than using a linear time scale). This act of simplification is a good trade-off between reducing the space complexity of the feature vectors while maintaining the relevant features of the music itself. 
After training, we will create an algorithm that can generate a stream of notes based on probability. 

### Data
We have gathered a dataset of over 400 MIDI files related to Bach's compositions, and have already pre-processed them. We decided to transpose each composition into the same key, in which we wrote a script to do so. Basically, this means that we modified each of the key signatures to be either C if the piece was major or A minor if the piece was minor. We also filtered out compositions that did not have the same time signature (we chose the time signatures 3/4 and 4/4 for regularity).

## The Models

### LSTM (Long Short-Term Recurrent Neural Network)
For the LSTM, after the data has been transposed it is converted into a single string with spaces marking the end of each 16th note and words representing different notes being played at once with a characters representing each note. These strings are fed into our recurrent neural net. Once the RNN is trained we first select a character and then generate an output distribution based on the current character and history and choose the next character based on that and continue this process for some number of characters. Then the string of characters is converted back into a midi file. 
Though we have finished the first part of creating the network and converting from midi to a text form we hope will be conducive to learning we still need to tune the parameter and train for a longer amount of time before we can fully know how successful this model is. Furthermore we hope to try a vector instead of character based rnn to see which one works better. 


### GAN (Generative Adversial Network)




