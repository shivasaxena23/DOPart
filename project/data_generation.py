import numpy as np

def system_values(i):

    if i == 0:
        with open('resnet34_compute_values_224_t4.npy', 'rb') as f:
            model_compute_values = np.load(f,allow_pickle=True)
        model_compute_values_remote = model_compute_values[1000:10000,:] #1000 or 10000
        compute_values_remote = np.mean(model_compute_values_remote,axis=0)
        input_data_real = np.array([224*224*3, 64*112*112, 64*56*56, 64*56*56, 64*56*56, 64*56*56, 64*56*56, 64*56*56, 64*56*56, 128*28*28, 128*28*28, 128*28*28, 128*28*28, 128*28*28, 128*28*28, 128*28*28, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 256*14*14, 512*7*7, 512*7*7, 512*7*7, 512*7*7, 512*7*7, 512*1*1])
    else:
        compute_values_remote = np.concatenate([np.zeros(i-1), [10]])
        input_data_real = np.concatenate([[10]*i]) 
    print(len(compute_values_remote),len(input_data_real))
    return compute_values_remote,input_data_real