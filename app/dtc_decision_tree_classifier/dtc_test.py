## IMPORT ##########

import os
import yaml
import numpy
import random
import sqlite3
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn import tree

####################

def main() :

    MAIN_PATH = os.getenv("MINMAX_ODM_MAINPATH").replace("\r", "").replace("\n", "")
    STATS_DATASET_PATH = MAIN_PATH + "/data/dataset_custom/dataset_stats/"
    SAVE_PATH = MAIN_PATH+"/data/dtc_decision_tree_classifier"
    
    configfile = MAIN_PATH + "/conf/config.yaml"
    with open(configfile, 'r') as ymlfile :
        cfg = yaml.load(ymlfile, yaml.loader.SafeLoader)
        ymlfile.close()

    ALARM_TYPE = cfg["global_parameters"]["alarm_type"]
    RANDOM_SEED = cfg["global_parameters"]["random_seed"]
    
    T_mult = cfg["dataset"]["T_mult"]
    F_mult = cfg["dataset"]["F_mult"]

    if RANDOM_SEED != 0 : random.seed(RANDOM_SEED)

    S_features = ["abs_S_Smin","rel_S_Smin_semi_width","rel_S_Smin_full_width","count_anomalies_S","ratio_anomalies_S","max_variation_S"]
    T_features = ["abs_T_Tmin","rel_T_Tmin_semi_width","rel_T_Tmin_full_width","count_anomalies_T","ratio_anomalies_T","max_variation_T"]
    B_features = ["mean_correlation","nb_measurements"]

    feature_names=numpy.array(S_features+T_features+B_features)

    abs_S_Smin = cfg["features"]["abs_S_Smin"]
    rel_S_Smin_semi_width = cfg["features"]["rel_S_Smin_semi_width"]
    rel_S_Smin_full_width = cfg["features"]["rel_S_Smin_full_width"]
    count_anomalies_S = cfg["features"]["count_anomalies_S"]
    ratio_anomalies_S = cfg["features"]["ratio_anomalies_S"]
    max_variation_S = cfg["features"]["max_variation_S"]
    abs_T_Tmin = cfg["features"]["abs_T_Tmin"]
    rel_T_Tmin_semi_width = cfg["features"]["rel_T_Tmin_semi_width"]
    rel_T_Tmin_full_width = cfg["features"]["rel_T_Tmin_full_width"]
    count_anomalies_T = cfg["features"]["count_anomalies_T"]
    ratio_anomalies_T = cfg["features"]["ratio_anomalies_T"]
    max_variation_T = cfg["features"]["max_variation_T"]
    mean_correlation = cfg["features"]["mean_correlation"]
    nb_measurements = cfg["features"]["nb_measurements"]

    S_features_filter = [abs_S_Smin,rel_S_Smin_semi_width,rel_S_Smin_full_width,count_anomalies_S,ratio_anomalies_S,max_variation_S]
    T_features_filter = [abs_T_Tmin,rel_T_Tmin_semi_width,rel_T_Tmin_full_width,count_anomalies_T,ratio_anomalies_T,max_variation_T]
    B_features_filter = [mean_correlation,nb_measurements]

    feature_filter=numpy.array(S_features_filter+T_features_filter+B_features_filter)
    feature_names=feature_names[feature_filter]

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
        TrainLabel.extend([1 for i in range(train_size)])
    
    TestSet.extend(FalseAlarm[train_size:])
    TestLabel.extend([0 for i in range(test_size)])
    for i in range(F_mult) :
        TrainSet.extend(FalseAlarm[:train_size])
        TrainLabel.extend([0 for i in range(train_size)])
    
    criterion = cfg["dtc_decision_tree_classifier"]["architecture"]["criterion"]
    max_depth = cfg["dtc_decision_tree_classifier"]["architecture"]["max_depth"]
    min_samples_split = cfg["dtc_decision_tree_classifier"]["architecture"]["min_samples_split"]
    min_samples_leaf = cfg["dtc_decision_tree_classifier"]["architecture"]["min_samples_leaf"]
    max_features = cfg["dtc_decision_tree_classifier"]["architecture"]["max_features"]
    max_leaf_nodes = cfg["dtc_decision_tree_classifier"]["architecture"]["max_leaf_nodes"]
    min_impurity_decrease = cfg["dtc_decision_tree_classifier"]["architecture"]["min_impurity_decrease"]

    dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease) 
    dtc.fit(TrainSet, TrainLabel)
    TestPrediction = dtc.predict(TestSet)

    accuracy = accuracy_score(TestLabel, TestPrediction)
    precision = precision_score(TestLabel, TestPrediction)
    recall = recall_score(TestLabel, TestPrediction)
    f1score = f1_score(TestLabel, TestPrediction)
    f3score = fbeta_score(TestLabel, TestPrediction, beta=3)

    text_file = open(MAIN_PATH+"/data/dtc_decision_tree_classifier/performances.txt", "w")
    text_file.write("ROUNDED :\nAccuracy: %.1f\nPrecision: %.1f\nRecall: %.1f\nF1-score: %.1f\nF3-score: %.1f\n\n\n"%(100*accuracy, 100*precision, 100*recall, 100*f1score, 100*f3score))
    text_file.write("PRECISE :\nAccuracy: %f\nPrecision: %f\nRecall: %f\nF1-score: %f\nF3-score: %f"%(accuracy, precision, recall, f1score, f3score))
    text_file.close()
    
    plt.figure(figsize=(60, 40))
    tree.plot_tree(dtc, feature_names=feature_names, impurity=True, rounded=True)
    if SAVE_PATH+"/tree.png" in os.listdir(SAVE_PATH) :
        os.remove(SAVE_PATH+"/tree.png")
    plt.savefig(SAVE_PATH+"/tree.png")

    conn = sqlite3.connect(MAIN_PATH+"/data/database/my_database.db")
    cursor = conn.cursor()
    cursor.execute("""SELECT id FROM all_run WHERE M_type LIKE 'DTC' ORDER BY id DESC""")
    tab = cursor.fetchall()
    if len(tab) > 0 :
        id = tab[0][0] + 1
    else :
        id = 0
    cursor.execute("""INSERT INTO all_run VALUES (%i, 'DTC', '%s', %i, %i, %i)"""%(id, ALARM_TYPE, RANDOM_SEED, T_mult, F_mult))
    conn.commit()
    cursor.execute("""INSERT INTO all_performances VALUES (%i, 'DTC', %f, %f, %f, %f, %f)"""%(id, accuracy, precision, recall, f1score, f3score))
    conn.commit()
    cursor.execute("""INSERT INTO all_features_usage VALUES (%i,'DTC',%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g)"""%tuple([id]+list(feature_filter)))
    conn.commit()
    cursor.execute("""INSERT INTO dtc_parameters VALUES (%i,'DTC','%s',%d,%d,%d,%d,%d,%f)"""%(id,criterion,max_depth,min_samples_split,min_samples_leaf,max_features,max_leaf_nodes,min_impurity_decrease))
    conn.commit()

if __name__ == "__main__" :
    main()