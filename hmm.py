import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from vector_quantization import VQ_LGB
from hmmlearn import hmm


def preprocess_data(data, resample_interval=0.02, window_size=0.16, window_step=0.08):
    n = len(data)

    x  = [e[0] for e in data]
    y  = [e[1] for e in data]
    dt = [e[2] for e in data]
    dt = [e - min(dt) for e in dt]

    # resample data at every resample_interval
    resample_points = []
    sample_point = 0.0

    while sample_point < dt[-1]:
        resample_points.append(sample_point)

        sample_point += resample_interval

    new_x = np.interp(resample_points, dt, x)
    new_y = np.interp(resample_points, dt, y)

    # scale by creating a bounding_box between 0 and 1
    new_x = [(x-min(new_x)) / (max(new_x)-min(new_x)) for x in new_x]
    new_y = [(y-min(new_y)) / (max(new_y)-min(new_y)) for y in new_y]

    length = len(new_x)

    # compute the N-point DFT
    N    = int(window_size / resample_interval)
    step = int(window_step / resample_interval)
    iterator = 0

    coef_fft_x = []
    coef_fft_y = []

    while iterator+N <= length:
        x = new_x[iterator: iterator+N]
        y = new_y[iterator: iterator+N]

        coef_fft_x.append(np.fft.hfft(x, N))
        coef_fft_y.append(np.fft.hfft(y, N))

        iterator += step

    return coef_fft_x, coef_fft_y


def get_features(coef_fft, codebook, lenghts):
    features = [[None for _ in range(l)] for l in lenghts]

    for i in range(len(lenghts)):
        for j in range(lenghts[i]):
            minimum = 10

            for v in codebook:
                norm = np.linalg.norm(coef_fft[sum(lenghts[:i]) + j] - v)

                if norm < minimum:
                    minimum = norm
                    features[i][j] = v

    return features


# read the data
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

symbols = ['circle', 'infinity', 'left_arrow', 'right_arrow', 'square']
data_sets = ['train', 'validate', 'test']

data = {symbol: {data_set: {} for data_set in data_sets} for symbol in symbols}

for symbol in symbols:
    for data_set in data_sets:
        filename = f'{THIS_FOLDER}/symbol_dataset/{symbol}_{data_set}.csv'

        with open(filename) as file:
            reader = csv.reader(file, delimiter=',')
            line_count = 0

            for row in reader:
                if line_count != 0:
                    i = int(float(row[0]))

                    if i not in data[symbol][data_set]:
                        data[symbol][data_set][i] = []

                    data[symbol][data_set][i].append((float(row[1]), float(row[2]), float(row[3])))

                line_count += 1


# preprocess data
features_train = {symbol: [] for symbol in symbols}
lenghts_train  = {symbol: [] for symbol in symbols}
features_test  = {symbol: [] for symbol in symbols}
lenghts_test   = {symbol: [] for symbol in symbols}
codebook = {symbol: {'x': [], 'y': []} for symbol in symbols}

for symbol in symbols:
    coef_fft_x = []
    coef_fft_y = []

    for data_set in ['train', 'validate']:
        for i in data[symbol][data_set]:          
            result = preprocess_data(data[symbol][data_set][i])

            lenghts_train[symbol].append(len(result[0]))

            coef_fft_x += result[0]
            coef_fft_y += result[1]

    vq_lg = VQ_LGB(coef_fft_x, 256, 0.00005, 3000)
    vq_lg.run()
    codebook[symbol]['x'] = vq_lg.get_codebook()

    vq_lg = VQ_LGB(coef_fft_y, 256, 0.00005, 3000)
    vq_lg.run()
    codebook[symbol]['y'] = vq_lg.get_codebook()

    features_x = get_features(coef_fft_x, codebook[symbol]['x'], lenghts_train[symbol])
    features_y = get_features(coef_fft_y, codebook[symbol]['y'], lenghts_train[symbol])

    len_features = len(features_x)

    for i in range(len_features):
        for j in range(lenghts_train[symbol][i]):
            features_train[symbol].append(np.append(features_x[i][j], features_y[i][j]))


    coef_fft_x = []
    coef_fft_y = []

    for i in data[symbol]['test']:          
        result = preprocess_data(data[symbol]['test'][i])

        lenghts_test[symbol].append(len(result[0]))

        coef_fft_x += result[0]
        coef_fft_y += result[1]

    features_x = get_features(coef_fft_x, codebook[symbol]['x'], lenghts_test[symbol])
    features_y = get_features(coef_fft_y, codebook[symbol]['y'], lenghts_test[symbol])

    len_features = len(features_x)

    for i in range(len_features):
        for j in range(lenghts_test[symbol][i]):
            features_test[symbol].append(np.append(features_x[i][j], features_y[i][j]))


# train models
ergodic_4states_hmm = [hmm.GaussianHMM(n_components=4, covariance_type='full').fit(features_train[symbol], lenghts_train[symbol]) for symbol in symbols]
ergodic_8states_hmm = [hmm.GaussianHMM(n_components=8, covariance_type='full').fit(features_train[symbol], lenghts_train[symbol]) for symbol in symbols]
bakis_4states_hmm   = [hmm.GaussianHMM(n_components=4, covariance_type='diag', init_params='cm', params='cmt').fit(features_train[symbol], lenghts_train[symbol]) for symbol in symbols]
bakis_8states_hmm   = [hmm.GaussianHMM(n_components=8, covariance_type='diag', init_params='cm', params='cmt').fit(features_train[symbol], lenghts_train[symbol]) for symbol in symbols]

hmm_types = ['ergodic', 'bakis']
no_states = ['4states', '8states']
hmms = {'ergodic': {'4states': ergodic_4states_hmm, '8states': ergodic_8states_hmm}, 'bakis': {'4states': bakis_4states_hmm, '8states': bakis_8states_hmm}}

invervals_test = {symbol: [sum(lenghts_test[symbol][:i]) for i in range(len(lenghts_test[symbol])+1)] for symbol in symbols}
confusion_matrix = {hmm_type: {states: {actual_symbol: {pred_symbol: 0 for pred_symbol in symbols} for actual_symbol in symbols} for states in no_states} for hmm_type in hmm_types}

for hmm_type in hmm_types:
    for states in no_states:
        for feature_symbol in symbols:
            for i in range(len(invervals_test[feature_symbol])-1):
                score_list = [model.score(features_test[feature_symbol][invervals_test[feature_symbol][i]:invervals_test[feature_symbol][i+1]]) for model in hmms[hmm_type][states]]
                result = symbols[score_list.index(max(score_list))]
                confusion_matrix[hmm_type][states][feature_symbol][result] += 1

        print(hmm_type, states)
        for symbol in symbols:
            print(symbol, confusion_matrix[hmm_type][states][symbol])
        print()