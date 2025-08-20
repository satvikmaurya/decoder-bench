import numpy as np
import pandas as pd
import os
import pickle

from sklearn.linear_model import LinearRegression

# Script to use data from existing IBM systems to train a regression model to predict the noise of a gate given it's duration

# Prepare the dataset: IBM data to be downloaded into the given path
def extract_noise_latency(path='../../IBM Calibration Data/'):
    files = os.listdir(path)
    errors = ['Readout Error', 'CNOT Error']
    latencies = ['Readout Duration', 'CNOT Duration']
    data = {}
    for file in files:
        df = pd.read_csv(path + file)
        # print(df.keys())
        for error in errors:
            key = None
            if error == 'Readout Error':
                key = 'Readout assignment error '
            else:
                key = 'CNOT error '
                if key not in list(df.keys()):
                    key = 'ECR error '
            if error not in data.keys():
                data[error] = df[key].to_list()
            else:
                data[error] += df[key].to_list()
            pass
        for latency in latencies:
            key = None
            if latency == 'Readout Duration':
                key = 'Readout length (ns)'
            else:
                key = 'Gate time (ns)'
            if latency not in data.keys():
                data[latency] = df[key].to_list()
            else:
                data[latency] += df[key].to_list()
            pass

    removals = []
    for categ in ['CNOT Duration', 'CNOT Error']:
        d = data[categ]
        new_list = []
        for i, elem in enumerate(d):
            # First a delimited sequence (;)
            temp = elem.split(';')
            for t in temp:
                x = t.split(':')
                if len(x) < 2:
                    continue
                if int(float(x[1])) == 1:
                    removals.append(i)
                    continue
                new_list.append(float(x[1]))
            pass
        data[categ] = new_list
        pass

    # Delete extra entries
    for i in removals:
        del data['CNOT Duration'][i]

    readout = ['Readout Error', 'Readout Duration']
    cnot = ['CNOT Error', 'CNOT Duration']

    x = {}
    y = {}
    for r in readout:
        x[r] = data[r]
    for c in cnot:
        y[c] = data[c]

    df1 = pd.DataFrame().from_dict(x)
    df2 = pd.DataFrame().from_dict(y)

    df1.to_csv('data/readout_data.csv')
    df2.to_csv('data/cnot_data.csv')

    return

def train_models(path='./data/'):
    # X: gate latency/duration
    # y: gate error
    for op in ['readout', 'cnot']:
        model = LinearRegression()
        data = pd.read_csv(path + op + '_data.csv')
        X, y = None, None
        for key in data.keys():
            if 'Error' in key:
                y = np.array(data[key])
            if 'Duration' in key:
                X = np.array(data[key]).reshape(-1, 1)
        model.fit(X, y)
        with open('%s_model.bin'%(path + op), 'wb') as file:
            pickle.dump(model, file)
    return

def extract_t1t2(path='../../../IBM Calibration Data/'):
    files = os.listdir(path) # Path where all IBM Calibration Data files are stored
    t1 = []
    t2 = []
    for file in files:
        df = pd.read_csv(path + file)
        t1 += list(df['T1 (us)'])
        t2 += list(df['T2 (us)'])
        pass
    data = {'T1':t1, 'T2':t2}
    df = pd.DataFrame().from_dict(data)
    df.to_csv('./data/T1_T2.csv')
    return

if __name__ == "__main__":
    # prepare_data()
    train_models()