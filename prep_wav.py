from scipy.io import wavfile
import argparse
import numpy as np
import random

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def save_wav_dont_flatten(name, data):
    wavfile.write(name, 44100, data.astype(np.float32))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm


def sliceOnMod(input_data, target_data, mod):
    # Split the data on a modulus.

    # type cast to an integer the modulus
    mod = int(mod)

    # Split the data into 100 pieces
    input_split = np.array_split(input_data, 100)
    target_split = np.array_split(target_data, 100)

    val_input_data = []
    val_target_data = []
    # Traverse the range of the indexes of the input signal reversed and pop every 5th for val
    for i in reversed(range(len(input_split))):
        if i%mod == 0:
            # Store the validation data
            val_input_data.append(input_split[i])
            val_target_data.append(target_split[i])
            # Remove the validation data from training
            input_split.pop(i)
            target_split.pop(i)

    # Flatten val_data down to one dimension and concatenate
    val_input_data = np.concatenate(val_input_data)
    val_target_data = np.concatenate(val_target_data)

    # Concatinate b back together
    training_input_data = np.concatenate(input_split)
    training_target_data = np.concatenate(target_split)
    return (training_input_data, training_target_data, val_input_data, val_target_data)

def sliceRandomPercentage(input_data, target_data, percentage):
    # Do a random split of the data by slicing the data into 100 pieces and randomly 
    # chosing some number of them.

    if percentage < 0 and percentage > 100:
        raise ValueError("Perentage must be between 0 and 100")

    # Split the data into 100 pieces
    input_split = np.array_split(input_data, 100)
    target_split = np.array_split(target_data, 100)
    validationChunks = int(percentage)

    # Skip the first entry because it has a different array size if the array can't be 
    # devided evenly which screws things
    selection = random.sample(range(1,99), validationChunks)

    val_input_data = []
    val_target_data = []
    # Store the randomly selected values in C
    for i, val in enumerate(selection):
        val_input_data.append(input_split[val])
        val_target_data.append(target_split[val])

    # Flatten val_data down to one dimension and concatenate
    val_input_data = np.concatenate(val_input_data)
    val_target_data = np.concatenate(val_target_data)

    # remove the validation selections from b
    for i in sorted(selection, reverse=True):
        input_split.pop(i)
        target_split.pop(i)

    # Concatinate b back together
    training_input_data = np.concatenate(input_split)
    training_target_data = np.concatenate(target_split)
    return (training_input_data, training_target_data, val_input_data, val_target_data)

def nonConditionedWavParse(args):
    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)
    test_in_rate, test_in_data = wavfile.read(args.test_in_file)
    test_out_rate, test_out_data = wavfile.read(args.test_out_file)

    clean_data = in_data.astype(np.float32).flatten()
    target_data = out_data.astype(np.float32).flatten()
    in_test = test_in_data.astype(np.float32).flatten()
    out_test = test_out_data.astype(np.float32).flatten()

    # If Desired Normalize the data
    if (args.normalize):
        clean_data = normalize(clean_data).reshape(len(clean_data),1)
        target_data = normalize(target_data).reshape(len(target_data),1)
        in_test = normalize(in_test).reshape(len(test_in_data),1)
        out_test = normalize(out_test).reshape(len(test_out_data),1)

    if args.random_split:
        in_train, out_train, in_val, out_val = sliceRandomPercentage(clean_data, target_data, args.random_split)
    else:
        # Split the data on a twenty percent mod
        in_train, out_train, in_val, out_val = sliceOnMod(clean_data, target_data, args.mod_split)

    save_wav(args.path + "/train/" + args.name + "-input.wav", in_train)
    save_wav(args.path + "/train/" + args.name + "-target.wav", out_train)

    save_wav(args.path + "/test/" + args.name + "-input.wav", in_test)
    save_wav(args.path + "/test/" + args.name + "-target.wav", out_test)

    save_wav(args.path + "/val/" + args.name + "-input.wav", in_val)
    save_wav(args.path + "/val/" + args.name + "-target.wav", out_val)

def getFileMap():
    ############   USER EDIT ##########################
    # Each key is the parameter for conditioning, and each value is a list
    # of two files, the in.wav and the out.wav for each parameter setting.

    file_map = {
      0.0 : [
          'Recordings/20211027_LPB1_000_Training_Clean.wav',
          'Recordings/20211027_LPB1_000_Training_Dirty.wav',
          'Recordings/20211027_LPB1_000_Test_Clean.wav',
          'Recordings/20211027_LPB1_000_Test_Dirty.wav',
      ],
      0.25 : [
          'Recordings/20211027_LPB1_025_Training_Clean.wav',
          'Recordings/20211027_LPB1_025_Training_Dirty.wav',
          'Recordings/20211027_LPB1_025_Test_Clean.wav',
          'Recordings/20211027_LPB1_025_Test_Dirty.wav',
      ],
      0.5 : [
          'Recordings/20211027_LPB1_050_Training_Clean.wav',
          'Recordings/20211027_LPB1_050_Training_Dirty.wav',
          'Recordings/20211027_LPB1_050_Test_Clean.wav',
          'Recordings/20211027_LPB1_050_Test_Dirty.wav',
      ],
      0.75 : [
          'Recordings/20211027_LPB1_075_Training_Clean.wav',
          'Recordings/20211027_LPB1_075_Training_Dirty.wav',
          'Recordings/20211027_LPB1_075_Test_Clean.wav',
          'Recordings/20211027_LPB1_075_Test_Dirty.wav',
      ],
      1.0 : [
          'Recordings/20211027_LPB1_100_Training_Clean.wav',
          'Recordings/20211027_LPB1_100_Training_Dirty.wav',
          'Recordings/20211027_LPB1_100_Test_Clean.wav',
          'Recordings/20211027_LPB1_100_Test_Dirty.wav',
      ]
    }

    ############   USER EDIT ##########################
    return file_map

def conditionedWavParse(args):
    '''
    This script processes multiple datasets to condition on a given value, such
    as Gain. It currently set up to handle 1 conditioned parameter.
    The processed "input" data will be a stereo wav file (2 channels), where the
    first channel is the audio data, and the second is the conditioned parameter.
    Each additional wav file will be concatenated to the previous for both
    training and validation/test audio.
    The processed "output" data will be a concatenated mono wav file.

    Note: Ensure to use the "is 2" (input size = 2) arg with dist_model_recnet.py
          on the conditioned data.

    Note: This is intended to be used with the colab conditioning training script.

    Note: Assumes all .wav files are mono, float32, no metadata
    '''
    # Load all datasets (assuming each is a float32 single channel wav file)    
    file_map = getFileMap()

    # Load and Preprocess Data ###########################################
    all_clean_train = np.array([[],[]]) # 2 channels of all (in audio, param)
    all_clean_val = np.array([[],[]]) # 2 channels of all (in audio, param)
    all_clean_test = np.array([[],[]]) # 2 channels of all (in audio, param)
    all_target_train = np.array([[]]) # 1 channels of all (out audio)
    all_target_val = np.array([[]]) # 1 channels of all (out audio)
    all_target_test = np.array([[]]) # 1 channels of all (out audio)

    for param in sorted(file_map.keys()):

        # Load and Preprocess Data ###########################################
        in_rate, in_data = wavfile.read(file_map[param][0])
        out_rate, out_data = wavfile.read(file_map[param][1])
        test_in_rate, test_in_data = wavfile.read(file_map[param][2])
        test_out_rate, test_out_data = wavfile.read(file_map[param][3])

        clean_data = in_data.astype(np.float32).flatten()
        target_data = out_data.astype(np.float32).flatten()
        in_test = test_in_data.astype(np.float32).flatten()
        out_test = test_out_data.astype(np.float32).flatten()

        # If Desired Normalize the data
        if (args.normalize):
            clean_data = normalize(clean_data).reshape(len(clean_data),1)
            target_data = normalize(target_data).reshape(len(target_data),1)
            in_test = normalize(in_test).reshape(len(test_in_data),1)
            out_test = normalize(out_test).reshape(len(test_out_data),1)

        if args.random_split:
            in_train, out_train, in_val, out_val = sliceRandomPercentage(clean_data, target_data, args.random_split)
        else:
            # Split the data on a twenty percent mod
            in_train, out_train, in_val, out_val = sliceOnMod(clean_data, target_data, args.mod_split)

        # Create the parameter arrays
        param_temp_train = np.array([param]*len(in_train))
        param_temp_val = np.array([param]*len(in_val))
        param_temp_test = np.array([param]*len(in_test))

        # Append the audio and paramters to the full data sets 
        all_clean_train = np.append(all_clean_train, np.array([in_train, param_temp_train]), axis=1)
        all_clean_val = np.append(all_clean_val, np.array([in_val, param_temp_val]), axis=1)
        all_clean_test = np.append(all_clean_test, np.array([in_test, param_temp_test]), axis=1)

        all_target_train = np.append(all_target_train, out_train)
        all_target_val = np.append(all_target_val, out_val)
        all_target_test = np.append(all_target_test, out_test)

    # Save the wav files 
    save_wav_dont_flatten(args.path + "/train/" + args.name + "-input.wav", all_clean_train.T)
    save_wav_dont_flatten(args.path + "/val/" + args.name + "-input.wav", all_clean_val.T)
    save_wav_dont_flatten(args.path + "/test/" + args.name + "-input.wav", all_clean_test.T)

    save_wav(args.path + "/train/" + args.name + "-target.wav", all_target_train)
    save_wav(args.path + "/val/" + args.name + "-target.wav", all_target_val)
    save_wav(args.path + "/test/" + args.name + "-target.wav", all_target_test)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script prepairs the data data to be trained'''
    )
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("test_in_file")
    parser.add_argument("test_out_file")
    parser.add_argument("--normalize", "-n", type=bool, default=False)
    parser.add_argument("--parameter", "-p", type=bool, default=False)
    parser.add_argument("name")
    parser.add_argument("--mod_split", '-ms', default=5, help="The default splitting mechanism. Splits the training and validation data on a 5 mod (or 20 percent).")
    parser.add_argument("--random_split", '-rs', type=float, default=None, help="By default, the training is split on a modulus. However, desingnating a percentage between 0 and 100 will create a random data split between the training and validatation sets.")
    parser.add_argument("--path", type=str, default="Data")

    args = parser.parse_args()

    if args.parameter:
        print("Parameterized Data")
        conditionedWavParse(args)

    else:
        print("Non-Parameterized Data")
        nonConditionedWavParse(args)
