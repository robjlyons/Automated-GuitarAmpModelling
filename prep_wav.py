from scipy.io import wavfile
import argparse
import numpy as np

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

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

    if percentage < 0 & percentage > 100:
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

def main(args):
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
        clean_data = normalize(clean_data).reshape(len(X_all),1)
        target_data = normalize(target_data).reshape(len(y_all),1)
        test_in = normalize(test_in).reshape(len(X_all),1)
        test_out = normalize(test_out).reshape(len(X_all),1)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script prepairs the data data to be trained'''
    )
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("target_in_file")
    parser.add_argument("target_out_file")
    parser.add_argument("--normalize", "-n", type=bool, default=False)
    parser.add_argument("name")
    parser.add_argument("--mod_split", 'ms', default=5, help="The default splitting mechanism. Splits the training and validation data on a 5 mod (or 20%).")
    parser.add_argument("--random_split", '-rs', default=None, help="By default, the training is split on a modulus. However, desingnating a percentage between 0 and 100 will create a random data split between the training and validatation sets.")
    parser.add_argument("--path", type=str, default="Data")


    args = parser.parse_args()
    main(args)
