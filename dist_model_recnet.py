%%writefile dist_model_recnet.py

import CoreAudioML.miscfuncs as miscfuncs
import numpy as np
import random
import CoreAudioML.training as training
import CoreAudioML.dataset as CAMLdataset
import CoreAudioML.networks as networks
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import os
from scipy.io.wavfile import write
import librosa
from torch.nn.utils import clip_grad_norm_

# ------------------- Data Augmentation ------------------- #

def augment_audio(audio, sample_rate):
    """
    Apply online data augmentation.
    :param audio: Input audio data as a PyTorch tensor.
    :param sample_rate: Sample rate of the audio.
    :return: Augmented audio data as a PyTorch tensor.
    """
    # Ensure audio is on the CPU for NumPy operations
    audio = audio.cpu().numpy()

    # Check signal length
    if audio.ndim > 1:
        print(f"Unexpected audio shape before augmentation: {audio.shape}")
        audio = audio[:, 0]  # Select first channel if multiple dimensions

    if len(audio) < 2048:  # Minimum length for FFT
        print(f"Skipping augmentation: Audio length too short ({len(audio)} samples)")
        return torch.tensor(audio).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Add random noise
    noise = np.random.normal(0, 0.005, audio.shape)
    augmented_audio = audio + noise

    # Apply pitch shift
    try:
        pitch_shift = np.random.uniform(-2, 2)  # Shift by up to 2 semitones
        augmented_audio = librosa.effects.pitch_shift(y=augmented_audio, sr=sample_rate, n_steps=pitch_shift)
    except Exception as e:
        print(f"Pitch shifting failed: {e}")
        return torch.tensor(audio).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert back to PyTorch tensor and move to the original device
    return torch.tensor(augmented_audio).to('cuda' if torch.cuda.is_available() else 'cpu')

def print_dataset_statistics(dataset, subset_name):
    """
    Calculate and print dataset statistics.
    :param dataset: The dataset object.
    :param subset_name: Name of the subset (e.g., "train", "val").
    """
    print(f"Statistics for {subset_name} dataset:")
    print(f"Number of samples: {len(dataset)}")
    #print(f"Sample rate: {dataset.fs}")
    #print(f"Duration (seconds): {len(dataset) / dataset.fs}")
    print(f"Mean: {np.mean(dataset)}")
    print(f"Standard deviation: {np.std(dataset)}")


def apply_augmentations(dataset, sample_rate):
    """
    Apply augmentations to the dataset.
    Process each segment individually to ensure compatibility with augmentation functions.
    """
    for subset_name, subset in dataset.subsets.items():
        print(f"Applying augmentations to {subset_name} dataset...")
        augmented_inputs = []
        augmented_targets = []

        # Loop through each segment
        for i in range(len(subset.data['input'])):
            input_segment = subset.data['input'][i]
            target_segment = subset.data['target'][i]

            augmented_input = input_segment.clone()  # Clone to ensure immutability is avoided
            augmented_target = target_segment.clone()

            # Apply augmentation to each channel
            for segment_idx in range(input_segment.shape[1]):
                augmented_input[:, segment_idx, 0] = augment_audio(input_segment[:, segment_idx, 0], sample_rate)
                augmented_target[:, segment_idx, 0] = augment_audio(target_segment[:, segment_idx, 0], sample_rate)

            augmented_inputs.append(augmented_input)
            augmented_targets.append(augmented_target)

        # Replace the original data with the augmented version
        subset.data['input'] = torch.stack(augmented_inputs)
        subset.data['target'] = torch.stack(augmented_targets)


# ------------------- Model Initialization ------------------- #

def init_model(save_path, args):
    """Initialize or load a model based on the configuration."""
    if miscfuncs.file_check('model.json', save_path) and args.load_model:
        print("Loading existing model...")
        model_data = miscfuncs.json_load('model', save_path)
        return networks.load_model(model_data)
    else:
        print("No saved model found, creating a new one...")
        network = networks.SimpleRNN(input_size=args.input_size, unit_type=args.unit_type,
                                      hidden_size=args.hidden_size, output_size=args.output_size,
                                      skip=args.skip_con, num_layers=args.num_layers,
                                      bidirectional=args.bidirectional)
        network.save_model('model', save_path)
        return network

# ------------------- Main Function ------------------- #

def main(args):
    """The main method creates the recurrent network, trains it, and carries out validation/testing."""
    start_time = time.time()

    # If a load_config argument was provided, construct the file path to the config file
    if args.load_config:
        # Load the configs and write them onto the args dictionary, this will add new args and/or overwrite old ones
        configs = miscfuncs.json_load(args.load_config, args.config_location)
        for parameters in configs:
            args.__setattr__(parameters, configs[parameters])

    if args.model == 'SimpleRNN':
        model_name = args.model + args.device + '_' + args.unit_type + '_hs' + str(args.hidden_size) + '_pre_' + args.pre_filt
    if args.pre_filt == 'A-Weighting':
        with open('Configs/' + 'b_Awght_mk2.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            args.pre_filt = list(reader)
            args.pre_filt = args.pre_filt[0]
            for item in range(len(args.pre_filt)):
                args.pre_filt[item] = float(args.pre_filt[item])
    elif args.pre_filt == 'high_pass':
        args.pre_filt = [-0.85, 1]
    elif args.pre_filt == 'None':
        args.pre_filt = None

    # Generate name of directory where results will be saved
    save_path = os.path.join(args.save_location, args.device + '-' + args.load_config)

    # Check if an existing saved model exists, and load it, otherwise create a new model
    network = init_model(save_path, args)

    # Enable CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    network.to(device)

    # Set up training optimizer + scheduler + loss functions and training info tracker
    optimiser = torch.optim.Adam(network.parameters(), lr=args.learn_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()  # Mixed-precision training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)
    loss_functions = training.LossWrapper(args.loss_fcns, args.pre_filt)
    train_track = training.TrainTrack()
    writer = SummaryWriter(os.path.join('TensorboardData', args.device))

    # Load dataset
    dataset = CAMLdataset.DataSet(data_dir='Data')

    dataset.create_subset('train', frame_len=22050)
    dataset.load_file(os.path.join('train', args.file_name), 'train')

    dataset.create_subset('val')
    dataset.load_file(os.path.join('val', args.file_name), 'val')
    dataset.fs = 44100  # Replace with actual sample rate if available

    print("Train dataset input shape:", dataset.subsets['train'].data['input'][0].shape)
    print("Train dataset target shape:", dataset.subsets['train'].data['target'][0].shape)

    # Apply augmentations
    apply_augmentations(dataset, dataset.fs)

    # Training loop
    init_time = time.time() - start_time + train_track['total_time'] * 3600
    network.save_state = True
    patience_counter = 0

    for epoch in range(train_track['current_epoch'] + 1, args.epochs + 1):
          print(f"Epoch: {epoch}")
          ep_start_time = time.time()
          network.train()

          # Initialize epoch loss
          total_loss = 0.0

          # Mixed-precision training
          with torch.cuda.amp.autocast(enabled=True):  # Use torch.cuda.amp.autocast without device_type
              for i in range(0, dataset.subsets['train'].data['input'][0].size(0), args.batch_size):
                  inputs = dataset.subsets['train'].data['input'][0][i:i + args.batch_size].to(device)
                  targets = dataset.subsets['train'].data['target'][0][i:i + args.batch_size].to(device)

                  optimiser.zero_grad()  # Reset gradients

                  # Forward pass
                  predictions = network(inputs)
                  loss = loss_functions(predictions, targets)

                  # Scale the loss and backpropagate
                  scaler.scale(loss).backward()

                  # Gradient clipping (optional for stability)
                  clip_grad_norm_(network.parameters(), max_norm=1.0)

                  # Optimizer step with scaler
                  scaler.step(optimiser)
                  scaler.update()

                  total_loss += loss.item() * inputs.size(0)  # Accumulate batch loss

          epoch_loss = total_loss / dataset.subsets['train'].data['input'][0].size(0)  # Average loss
          writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)

          # Validation
          if epoch % args.validation_f == 0:
              network.eval()
              with torch.no_grad():
                  val_output, val_loss = network.process_data(
                      dataset.subsets['val'].data['input'][0].to(device),
                      dataset.subsets['val'].data['target'][0].to(device), loss_functions, args.val_chunk
                  )
              scheduler.step(val_loss)
              print(f"Validation loss: {val_loss}")

              if val_loss < train_track['best_val_loss']:
                  patience_counter = 0
                  network.save_model('model_best', save_path)
              else:
                  patience_counter += 1

              writer.add_scalar('Validation/Loss', val_loss.item(), epoch)

          # Log epoch metrics
          writer.add_scalar('Training/Loss', epoch_loss, epoch)
          writer.add_scalar('Training/LearningRate', optimiser.param_groups[0]['lr'], epoch)

          if patience_counter >= args.validation_p:
              print(f"Early stopping triggered at epoch {epoch}")
              break

          print(f"Epoch {epoch} complete. Loss: {epoch_loss}")


    print("Training complete. Running tests...")

    # Test the model
    dataset.create_subset('test')
    dataset.load_file(os.path.join(args.data_location, 'test', args.file_name), 'test')
    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                                  dataset.subsets['test'].data['target'][0], loss_functions,
                                                  args.test_chunk)
    writer.add_scalar('Testing/Loss', test_loss.item())

    print(f"Final test loss: {test_loss.item()}")


if __name__ == "__main__":
    prsr = argparse.ArgumentParser(
        description='''This script implements training for neural network amplifier/distortion effects modelling. This is
        intended to recreate the training of models of the ht1 amplifier and big muff distortion pedal, but can easily be 
        adapted to use any dataset''')

    # arguments for the training/test data locations and file names and config loading
    prsr.add_argument('--device', '-p', default='ht1', help='This label describes what device is being modelled')
    prsr.add_argument('--data_location', '-dl', default='..', help='Location of the "Data" directory')
    prsr.add_argument('--file_name', '-fn', default='ht1',
                      help='The filename of the wav file to be loaded as the input/target data, the script looks for files'
                           'with the filename and the extensions -input.wav and -target.wav ')
    prsr.add_argument('--load_config', '-l',
                      help="File path, to a JSON config file, arguments listed in the config file will replace the defaults"
                      , default='RNN3')
    prsr.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    prsr.add_argument('--save_location', '-sloc', default='Results', help='Directory where trained models will be saved')
    prsr.add_argument('--load_model', '-lm', default=True, help='load a pretrained model if it is found')
    prsr.add_argument('--seed', default=None, type=int, help='seed all of the random number generators if desired')

    # pre-processing of the training/val/test data
    prsr.add_argument('--segment_length', '-slen', type=int, default=22050, help='Training audio segment length in samples')

    # number of epochs and validation
    prsr.add_argument('--epochs', '-eps', type=int, default=2000, help='Max number of training epochs to run')
    prsr.add_argument('--validation_f', '-vfr', type=int, default=2, help='Validation Frequency (in epochs)')
    # TO DO
    prsr.add_argument('--validation_p', '-vp', type=int, default=25,
                      help='How many validations without improvement before stopping training, None for no early stopping')

    # settings for the training epoch
    prsr.add_argument('--batch_size', '-bs', type=int, default=50, help='Training mini-batch size')
    prsr.add_argument('--iter_num', '-it', type=int, default=None,
                      help='Overrides --batch_size and instead sets the batch_size so that a total of --iter_num batches'
                           'are processed in each epoch')
    prsr.add_argument('--learn_rate', '-lr', type=float, default=0.005, help='Initial learning rate')
    prsr.add_argument('--init_len', '-il', type=int, default=200,
                      help='Number of sequence samples to process before starting weight updates')
    prsr.add_argument('--up_fr', '-uf', type=int, default=1000,
                      help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                           'default argument updates every 1000 samples')
    prsr.add_argument('--cuda', '-cu', default=1, help='Use GPU if available')

    # loss function/s
    prsr.add_argument('--loss_fcns', '-lf', default={'ESRPre': 0.75, 'DC': 0.25},
                      help='Which loss functions, ESR, ESRPre, DC. Argument is a dictionary with each key representing a'
                           'loss function name and the corresponding value being the multiplication factor applied to that'
                           'loss function, used to control the contribution of each loss function to the overall loss ')
    prsr.add_argument('--pre_filt',   '-pf',   default='high_pass',
                        help='FIR filter coefficients for pre-emphasis filter, can also read in a csv file')

    # the validation and test sets are divided into shorter chunks before processing to reduce the amount of GPU memory used
    # you can probably ignore this unless during training you get a 'cuda out of memory' error
    prsr.add_argument('--val_chunk', '-vs', type=int, default=100000, help='Number of sequence samples to process'
                                                                                   'in each chunk of validation ')
    prsr.add_argument('--test_chunk', '-tc', type=int, default=100000, help='Number of sequence samples to process'
                                                                                   'in each chunk of validation ')

    # arguments for the network structure
    prsr.add_argument('--model', '-m', default='SimpleRNN', type=str, help='model architecture')
    prsr.add_argument('--input_size', '-is', default=1, type=int, help='1 for mono input data, 2 for stereo, etc ')
    prsr.add_argument('--output_size', '-os', default=1, type=int, help='1 for mono output data, 2 for stereo, etc ')
    prsr.add_argument('--num_blocks', '-nb', default=1, type=int, help='Number of recurrent blocks')
    prsr.add_argument('--hidden_size', '-hs', default=16, type=int, help='Recurrent unit hidden state size')
    prsr.add_argument('--unit_type', '-ut', default='LSTM', help='LSTM or GRU or RNN')
    prsr.add_argument('--skip_con', '-sc', default=1, help='is there a skip connection for the input to the output')
    prsr.add_argument('--num_layers', '-nl', default=1, type=int, help="Number of RNN/Transformer layers.")
    prsr.add_argument('--bidirectional', '-bd', type=bool, default=True,
                        help="Enable bidirectional RNNs (applicable for LSTM/GRU/RNN).")

    args = prsr.parse_args()
    main(args)
