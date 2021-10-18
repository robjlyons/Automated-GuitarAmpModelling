from scipy.io import wavfile
import argparse
import numpy as np

def getFileMap():
    ############   USER EDIT ##########################
    # Each key is the parameter for conditioning, and each value is a list
    # of two files, the in.wav and the out.wav for each parameter setting.

    file_map = {
      0.0 : ['blackstar_ht40_clean_in.wav', 'blackstar_ht40_clean_out.wav'],
      0.25 : ['blackstar_ht40_g25_in.wav', 'blackstar_ht40_g25_out.wav'],
      0.5 : ['blackstar_ht40_g5_in.wav', 'blackstar_ht40_g5_out.wav'],
      0.75 : ['blackstar_ht40_g75_in.wav', 'blackstar_ht40_g75_out.wav'],
      1.0 : ['blackstar_ht40_gain10_in.wav', 'blackstar_ht40_gain10_out.wav']
    }

    ############   USER EDIT ##########################
    return file_map

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def save_combined_wav(name, data):
    # Save two channel wav file with all audio concatenated
    wavfile.write(name, 44100, data.astype(np.float32))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def main(args):
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
    all_x_train = np.array([[],[]]) # 2 channels of all (in audio, param)
    all_x_val = np.array([[],[]]) # 2 channels of all (in audio, param)
    all_y_train = np.array([[]]) # 1 channels of all (out audio)
    all_y_val = np.array([[]]) # 1 channels of all (out audio)

    for param in sorted(file_map.keys()):
      in_rate, in_data = wavfile.read(file_map[param][0])
      out_rate, out_data = wavfile.read(file_map[param][1])

      X_all = in_data.astype(np.float32).flatten()   
      X_all = normalize(X_all).flatten()  
      y_all = out_data.astype(np.float32).flatten()  
      y_all = normalize(y_all).flatten() 

      # Get the last 20% of the wav data to run prediction and plot results
      y_train, y_val = np.split(y_all, [int(len(y_all)*.8)])
      x_train, x_val= np.split(X_all, [int(len(X_all)*.8)])

      # Append the audio data to the "all" arrays, add param data to x arrays for conditioning
      param_temp_train = np.array([param]*len(x_train))
      param_temp_val = np.array([param]*len(x_val))

      all_x_train = np.append(all_x_train, np.array([x_train, param_temp_train]) , axis=1)
      all_x_val = np.append(all_x_val, np.array([x_val, param_temp_val]) , axis=1)

      all_y_train = np.append(all_y_train, y_train)
      all_y_val = np.append(all_y_val, y_val)

    save_combined_wav(args.path + "/train/" + args.name + "-input.wav", all_x_train.T)
    save_wav(args.path + "/train/" + args.name + "-target.wav", all_y_train)

    #NOTE: The proper way to conduct training would be to have a different
    #      dataset for testing, so that the model tests on audio it hasn't
    #      seen during training. I resused the validation data here due to
    #      limited available audio data. 
    save_combined_wav(args.path + "/test/" + args.name + "-input.wav", all_x_val.T)
    save_wav(args.path + "/test/" + args.name + "-target.wav", all_y_val)

    save_combined_wav(args.path + "/val/" + args.name + "-input.wav", all_x_val.T) 
    save_wav(args.path + "/val/" + args.name + "-target.wav", all_y_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--path", type=str, default="Data")
    args = parser.parse_args()
    main(args)