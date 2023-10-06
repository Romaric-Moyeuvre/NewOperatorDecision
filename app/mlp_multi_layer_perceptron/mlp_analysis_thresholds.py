## IMPORT ##########

import os
import math
import yaml
import numpy
import random
import sqlite3
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix

import torch
import torch.nn as nn

####################

def main() :

    MAIN_PATH = os.getenv("MINMAX_ODM_MAINPATH").replace("\r", "").replace("\n", "")
    STATS_DATASET_PATH = MAIN_PATH + "/data/dataset_custom/dataset_stats/"
    SAVE_PATH = MAIN_PATH+"/data/mlp_multi_layer_perceptron"
    
    configfile = MAIN_PATH + "/conf/config.yaml"
    with open(configfile, 'r') as ymlfile :
        cfg = yaml.load(ymlfile, yaml.loader.SafeLoader)
        ymlfile.close()

    ALARM_TYPE = cfg["global_parameters"]["alarm_type"]
    RANDOM_SEED = cfg["global_parameters"]["random_seed"]
    
    T_mult = cfg["dataset"]["T_mult"]
    F_mult = cfg["dataset"]["F_mult"]

    if RANDOM_SEED != 0 : random.seed(RANDOM_SEED)

    S_features = ["abs_S_Smin","rel_S_Smin_semi_width","rel_S_Smin_full_width","abs_S_Smax","rel_S_Smax_semi_width","rel_S_Smax_full_width","count_anomalies_S","ratio_anomalies_S","max_variation_S"]
    T_features = ["abs_T_Tmin","rel_T_Tmin_semi_width","rel_T_Tmin_full_width","abs_T_Tmax","rel_T_Tmax_semi_width","rel_T_Tmax_full_width","count_anomalies_T","ratio_anomalies_T","max_variation_T"]
    B_features = ["mean_correlation","nb_measurements"]

    feature_names=numpy.array(S_features+T_features+B_features)

    abs_S_Smin = cfg["features"]["abs_S_Smin"]
    rel_S_Smin_semi_width = cfg["features"]["rel_S_Smin_semi_width"]
    rel_S_Smin_full_width = cfg["features"]["rel_S_Smin_full_width"]
    abs_S_Smax = cfg["features"]["abs_S_Smax"]
    rel_S_Smax_semi_width = cfg["features"]["rel_S_Smax_semi_width"]
    rel_S_Smax_full_width = cfg["features"]["rel_S_Smax_full_width"]
    count_anomalies_S = cfg["features"]["count_anomalies_S"]
    ratio_anomalies_S = cfg["features"]["ratio_anomalies_S"]
    max_variation_S = cfg["features"]["max_variation_S"]
    abs_T_Tmin = cfg["features"]["abs_T_Tmin"]
    rel_T_Tmin_semi_width = cfg["features"]["rel_T_Tmin_semi_width"]
    rel_T_Tmin_full_width = cfg["features"]["rel_T_Tmin_full_width"]
    abs_T_Tmax = cfg["features"]["abs_T_Tmax"]
    rel_T_Tmax_semi_width = cfg["features"]["rel_T_Tmax_semi_width"]
    rel_T_Tmax_full_width = cfg["features"]["rel_T_Tmax_full_width"]
    count_anomalies_T = cfg["features"]["count_anomalies_T"]
    ratio_anomalies_T = cfg["features"]["ratio_anomalies_T"]
    max_variation_T = cfg["features"]["max_variation_T"]
    mean_correlation = cfg["features"]["mean_correlation"]
    nb_measurements = cfg["features"]["nb_measurements"]

    S_features_filter = [abs_S_Smin,rel_S_Smin_semi_width,rel_S_Smin_full_width,abs_S_Smax,rel_S_Smax_semi_width,rel_S_Smax_full_width,count_anomalies_S,ratio_anomalies_S,max_variation_S]
    T_features_filter = [abs_T_Tmin,rel_T_Tmin_semi_width,rel_T_Tmin_full_width,abs_T_Tmax,rel_T_Tmax_semi_width,rel_T_Tmax_full_width,count_anomalies_T,ratio_anomalies_T,max_variation_T]
    B_features_filter = [mean_correlation,nb_measurements]

    feature_filter=numpy.array(S_features_filter+T_features_filter+B_features_filter)
    feature_names=feature_names[feature_filter]

    growth_rate = cfg["mlp_multi_layer_perceptron"]["growth_rate"]
    dropout = cfg["mlp_multi_layer_perceptron"]["dropout"]
    learning_rate = cfg["mlp_multi_layer_perceptron"]["learning_rate"]
    batch_size = cfg["mlp_multi_layer_perceptron"]["batch_size"]
    epochs = cfg["mlp_multi_layer_perceptron"]["epochs"]
    factor = cfg["mlp_multi_layer_perceptron"]["factor"]
    patience = cfg["mlp_multi_layer_perceptron"]["patience"]
    epsilon = cfg["mlp_multi_layer_perceptron"]["epsilon"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class NeuralNetworkModel(nn.Module) :
        def __init__(self, input_size, growth_rate):
            super().__init__()
            self.blockA = nn.Sequential(
                nn.Linear(input_size, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU())
            self.blockB = nn.Sequential(
                nn.Linear(growth_rate, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU())
            self.blockC = nn.Sequential(
                nn.Linear(2*growth_rate, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU())
            self.blockD = nn.Sequential(
                nn.Linear(3*growth_rate, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU())
            self.blockE = nn.Sequential(
                nn.Linear(4*growth_rate, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU())
            self.blockF = nn.Sequential(
                nn.Linear(5*growth_rate, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU())
            self.line = nn.Linear(growth_rate, 1)
            self.sigm = nn.Sigmoid()
        def forward(self, z):
            outA = self.blockA(z)
            outB = self.blockB(outA)
            outC = self.blockC(torch.cat((outA, outB), dim=1))
            outD = self.blockD(torch.cat((outA, outB, outC), dim=1))
            outE = self.blockE(torch.cat((outA, outB, outC, outD), dim=1))
            outF = self.blockF(torch.cat((outA, outB, outC, outD, outE), dim=1))
            out = self.line(outF)
            out = self.sigm(out)
            return out

    class Dataset(torch.utils.data.Dataset) :
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __getitem__(self, index):
            x = self.data[index]
            y = self.labels[index]
            x = torch.from_numpy(numpy.array(x)).float()
            y = torch.from_numpy(numpy.array(y)).float()
            return x, y
        def __len__(self):
            return len(self.data)

    TrueAlarm = []
    FalseAlarm = []

    if ALARM_TYPE == "BOTH" or ALARM_TYPE == "TEMP" :
        path_temp_true = STATS_DATASET_PATH + "temperature/true_alarm"
        for file in os.listdir(path_temp_true) :
            if file.endswith(".npy") == True :
                TrueAlarm.append(numpy.load(path_temp_true+"/"+file)[feature_filter])
        path_temp_false = STATS_DATASET_PATH + "temperature/false_alarm"
        for file in os.listdir(path_temp_false) :
            if file.endswith(".npy") == True :
                FalseAlarm.append(numpy.load(path_temp_false+"/"+file)[feature_filter])

    if ALARM_TYPE == "BOTH" or ALARM_TYPE == "PSAL" :
        path_temp_true = STATS_DATASET_PATH + "salinity/true_alarm"
        for file in os.listdir(path_temp_true) :
            if file.endswith(".npy") == True :
                TrueAlarm.append(numpy.load(path_temp_true+"/"+file)[feature_filter])
        path_temp_false = STATS_DATASET_PATH + "salinity/false_alarm"
        for file in os.listdir(path_temp_false) :
            if file.endswith(".npy") == True :
                FalseAlarm.append(numpy.load(path_temp_false+"/"+file)[feature_filter])

    Lmean = []
    Lstd = []
    for i in range(len(TrueAlarm[0])) :
        values_i = [ta[i] for ta in TrueAlarm] + [fa[i] for fa in FalseAlarm]
        mean_i = numpy.mean(values_i)
        std_i = numpy.std(values_i)
        Lmean.append(mean_i)
        Lstd.append(std_i)

    for measure in TrueAlarm :
        for i, value in enumerate(measure) :
            measure[i] = (value - Lmean[i]) / Lstd[i]

    for measure in FalseAlarm :
        for i, value in enumerate(measure) :
            measure[i] = (value - Lmean[i]) / Lstd[i]

    TestSet, TestLabel, TrainSet, TrainLabel = [], [], [], []

    sample_size = min(len(TrueAlarm), len(FalseAlarm))

    indices_true = random.sample(range(len(TrueAlarm)), sample_size)
    indices_false = random.sample(range(len(FalseAlarm)), sample_size)

    TrueAlarm = numpy.array(TrueAlarm)
    FalseAlarm = numpy.array(FalseAlarm)
    
    TrueAlarm = TrueAlarm[indices_true]
    FalseAlarm = FalseAlarm[indices_false]

    train_size = int(sample_size * 0.75)
    test_size = sample_size - train_size

    TestSet.extend(TrueAlarm[train_size:])
    TestLabel.extend([1 for i in range(test_size)])
    for i in range(T_mult) :
        TrainSet.extend(TrueAlarm[:train_size])
        TrainLabel.extend([1.0 for i in range(train_size)])
    
    TestSet.extend(FalseAlarm[train_size:])
    TestLabel.extend([0 for i in range(test_size)])
    for i in range(F_mult) :
        TrainSet.extend(FalseAlarm[:train_size])
        TrainLabel.extend([0.0 for i in range(train_size)])
    
    test_dataset = Dataset(TestSet, TestLabel)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    train_dataset = Dataset(TrainSet, TrainLabel)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    net = NeuralNetworkModel(len(TrueAlarm[0]), growth_rate)
    net = net.to(device)
    net.dropout = nn.Dropout(dropout)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience, eps=epsilon)

    List_Loss = []

    for epoch in range(epochs) :
        total_loss = 0.0
        for data in train_loader :
            input, label = data
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label.reshape([label.shape[0], 1]))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step(total_loss/len(train_dataset))
        List_Loss.append(total_loss/len(train_dataset))

    labels_sklearn_metrics = []
    predictions_sklearn_metrics = []
    output_sklearn_metrics = []
    net.eval()
    with torch.no_grad() :
        correct = 0.0
        total = 0.0
        for data in test_loader :
            input, label = data
            input = input.to(device)
            label = label.to(device)
            output = net(input)
            predicted = torch.where(output >= 0.5, 1.0, 0.0)
            correct += (predicted == label).sum().item()
            total += predicted.nelement()
            output_sklearn_metrics.extend(output.tolist())
            labels_sklearn_metrics.extend(label.tolist())
            predictions_sklearn_metrics.extend(predicted.tolist())

    accuracy = accuracy_score(labels_sklearn_metrics, predictions_sklearn_metrics)
    precision = precision_score(labels_sklearn_metrics, predictions_sklearn_metrics)
    recall = recall_score(labels_sklearn_metrics, predictions_sklearn_metrics)
    f1score = f1_score(labels_sklearn_metrics, predictions_sklearn_metrics)
    f3score = fbeta_score(labels_sklearn_metrics, predictions_sklearn_metrics, beta=3)

    text_file = open(SAVE_PATH+"/performances.txt", "w")
    text_file.write("ROUNDED :\nAccuracy: %.1f\nPrecision: %.1f\nRecall: %.1f\nF1-score: %.1f\nF3-score: %.1f\n\n\n"%(100*accuracy, 100*precision, 100*recall, 100*f1score, 100*f3score))
    text_file.write("PRECISE :\nAccuracy: %f\nPrecision: %f\nRecall: %f\nF1-score: %f\nF3-score: %f"%(accuracy, precision, recall, f1score, f3score))
    text_file.close()
    
    plt.figure(figsize=(30, 20))
    plt.plot([epoch for epoch in range(epochs)], List_Loss)
    plt.savefig(SAVE_PATH+"/mlp_loss.png")

    conn = sqlite3.connect(MAIN_PATH+"/data/database/my_database.db")
    cursor = conn.cursor()
    cursor.execute("""SELECT id FROM all_run WHERE M_type LIKE 'MLP' ORDER BY id DESC""")
    tab = cursor.fetchall()
    if len(tab) > 0 :
        id = tab[0][0] + 1
    else :
        id = 0
    cursor.execute("""INSERT INTO all_run VALUES (%i, 'MLP', '%s', %i, %i, %i)"""%(id, ALARM_TYPE, RANDOM_SEED, T_mult, F_mult))
    conn.commit()
    cursor.execute("""INSERT INTO all_performances VALUES (%i, 'MLP', %f, %f, %f, %f, %f)"""%(id, accuracy, precision, recall, f1score, f3score))
    conn.commit()
    cursor.execute("""INSERT INTO all_features_usage VALUES (%i,'DTC',%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g)"""%tuple([id]+list(feature_filter)))
    conn.commit()
    cursor.execute("""INSERT INTO mlp_parameters VALUES (%i,'MLP',%d,%f,%f,%d,%f,%d,%d)"""%(id, growth_rate,dropout,learning_rate,epochs,factor,patience,int(math.log10(epsilon))))
    conn.commit()

    L_threshold = [0.01*x for x in range(101)]
    L_performances = []
    for threshold in L_threshold :
        output_sklearn_metrics = []
        labels_sklearn_metrics = []
        predictions_sklearn_metrics = []
        with torch.no_grad() :
            for data in test_loader :
                input, label = data
                input = input.to(device)
                label = label.to(device)
                output = net(input)
                predicted = torch.where(output >= threshold, 1.0, 0.0)
                output_sklearn_metrics.extend(output.tolist())
                labels_sklearn_metrics.extend(label.tolist())
                predictions_sklearn_metrics.extend(predicted.tolist())

        accuracy = accuracy_score(labels_sklearn_metrics, predictions_sklearn_metrics)
        precision = precision_score(labels_sklearn_metrics, predictions_sklearn_metrics)
        recall = recall_score(labels_sklearn_metrics, predictions_sklearn_metrics)
        f1score = f1_score(labels_sklearn_metrics, predictions_sklearn_metrics)
        f3score = fbeta_score(labels_sklearn_metrics, predictions_sklearn_metrics, beta=3)
        L_performances.append([accuracy, precision, recall, f1score, f3score])

    plt.figure(figsize=(30, 20))
    lineObjects = plt.plot(L_threshold, L_performances)
    plt.legend(iter(lineObjects), ('Accuracy', 'Precision', 'Recall', 'F1Score', 'F3score'))
    plt.savefig(SAVE_PATH+"/mlp_thresholds.png")

if __name__ == "__main__" :
    main()

