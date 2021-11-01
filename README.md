# Automated-Guitar Amplifier Modelling

## GuitarML Fork
This fork adds Colab Training, wav file pre-processing, results plotting, and some helpful printouts during training.
Use the example config "config/RNN3-acoustic1-pre.json" to train models for [NeuralPi](https://github.com/GuitarML/NeuralPi)<br>

Models for NeuralPi must have hidden_size=20 specified in the config file.

```
git clone https://github.com/GuitarML/Automated-GuitarAmpModelling.git
cd Automated-GuitarAmpModelling
git submodule update --init --recursive
```
```
# Add your input.wav and output.wav to the top directory, and add your config to the config directory.
# Name the .wav files and config file appropriately in the following command
python prep_wav.py input.wav output.wav acoustic1-pre
```
```
# Edit to use your config in the following command
# The model will be located in Results/
python dist_model_recnet.py -l "RNN3-acoustic1-pre"
```
```
# Edit your config name in the following command
# The plots will be generated in the Results/modelName/ directory
python plot.py acoustic1-pre
```

### Training Conditioned Models

Helper scipts have been added to aid in training conditioned models. These models are capable of reproducing the full range of a particular knob, such as Gain/Drive. The ```colab_conditioned_training.ipynb``` script can be used to train these types of models. The ```prep_wav_cond.py``` must first be edited to reference your wav files along with the value of the parameter for each set of in/out recordings. 

You will still use mono, 32FP wav files to start. After running the ```prep_wav_cond.py``` on these files, the processsed input wav files are now 2 channel (stereo), with the audio data on the first channel and the conditioning parameter on the second channel. For best results, record 4-5 samples of the full range of the gain/drive knob normalized as 0.0 to 1.0. For example: (0.0 0.33, 0.66, 1.0) or (0.0, 0.25, 0.5, 0.75, 1.0). 

```
# Edit the "file_map" dictionary variable in the following script to reference your wav files
python prep_wav_cond.py ht40cond

# The "-is 2" flag specifies that two inputs to the network are used, 1 for the audio data and 1 for the conditioning parameter
python dist_model_recnet.py -l "RNN3-ht40cond" -is 2
```

Note: [NeuralPi](https://github.com/GuitarML/NeuralPi) version 1.3 has the ability to run models conditioned on a single parameter as a real-time guitar plugin (the conditioned parameter is assumed to be gain/drive).

### Using Transfer Learning
You can greatly improve training by starting with a pre-trained model of a similar amp/pedal. Simply start and stop the training script to generate the model file in the "Results" folder, then replace the "model.json" file with another trained model and restart training. This gives the training a head start, and can also reduce the amount of training data needed for a accurate model. 

For more information on using Transfer Learning for Guitar Effects check out this article published on [Towards Data Science](https://towardsdatascience.com/transfer-learning-for-guitar-effects-4af50609dce1)

##

This repository contains neural network training scripts and trained models of guitar amplifiers and distortion pedals. The 'Results' directory contains some example recurrent neural network models trained to emulate the ht-1 amplifier and Big Muff Pi fuzz pedal, these models are described in this [conference paper](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)

## Using this repository
It is possible to use this repository to train your own models. To model a different distortion pedal or amplifier, a dataset recorded from your target device is required, example datasets recorded from the ht1 and Big Muff Pi are contained in the 'Data' directory. 

### Cloning this repository

To create a working local copy of this repository, use the following command:

git clone --recurse-submodules https://github.com/Alec-Wright/NeuralGuitarAmpModelling

### Python Environment

Using this repository requires a python environment with the 'pytorch', 'scipy', 'tensorboard' and 'numpy' packages installed. 
Additionally this repository uses the 'CoreAudioML' package, which is included as a submodule. Cloining the repo as described in 'Cloning this repository' ensures the CoreAudioML package is also downloaded.

### Processing Audio

The 'proc_audio.py' script loads a neural network model and uses it to process some audio, then saving the processed audio. This is also a good way to check if your python environment is setup correctly. Running the script with no arguments:

python proc_audio.py

will use the default arguments, the script will load the 'model_best.json' file from the directory 'Results/ht1-ht11/' and use it to process the audio file 'Data/test/ht1-input.wav', then save the output audio as 'output.wav'
Different arguments can be used as follows

python proc_audio.py 'path/to/input_audio.wav' 'output_filename.wav' 'Results/path/to/model_best.json'

### Training Script

the 'dist_model_recnet.py' script was used to train the example models in the 'Results' directory. At the top of the file the 'argparser' contains a description of all the training script arguments, as well as their default values. To train a model using the default arguments, simply run the model from the command line as follows:

python dist_model_recnet.py

note that you must run this command from a python environment that has the libraries described in 'Python Environment' installed. To use different arguments in the training script you can change the default arguments directly in 'dist_model_recnet.py', or you can direct the 'dist_model_recnet.py' script to look for a config file that contains different arguments, for example by running the script using the following command:

python dist_model_recnet.py -l "ht11.json"

Where in this case the script will look for the file ht11.json in the the 'Configs' directory. To create custom config files, the ht11.json file provided can be edited in any text editor.

During training, the script will save some data to a folder in the Results directory. These are, the lowest loss achieved on the validation set so far in 'bestvloss.txt', as well as a copy of that model 'model_best.json', and the audio created by that model 'best_val_out.wav'. The neural network at the end of the most recent training epoch is also saved, as 'model.json'. When training is complete the test dataset is processed, and the audio produced and the test loss is also saved to the same directory.

A trained model contained in one of the 'model.json' or 'model_best.json' files can be loaded, see the 'proc_audio.py' script for an example of how this is done.

### Feedback

This repository is still a work in progress, and I welcome your feedback! Either by raising an Issue or in the 'Discussions' tab 
