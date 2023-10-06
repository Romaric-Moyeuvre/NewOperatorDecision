## IMPORT ##########

import os
import sqlite3

####################

def main() :

    MAIN_PATH = os.getenv("MINMAX_ODM_MAINPATH").replace("\r", "").replace("\n", "")

    file = open(MAIN_PATH+"/data/database/my_database.db", 'w')
    file.close()

    conn = sqlite3.connect(MAIN_PATH+"/data/database/my_database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS all_run (
            id INTEGER NOT NULL,
            M_type TEXT NOT NULL,
            A_type TEXT NOT NULL,
            seed INTEGER NOT NULL,
            T_mult INTEGER NOT NULL,
            F_mult INTEGER NOT NULL 
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS all_performances (
            id INTEGER NOT NULL,
            M_type TEXT NOT NULL,
            accuracy FLOAT NOT NULL,
            precision FLOAT NOT NULL,
            recall FLOAT NOT NULL,
            f1_score FLOAT NOT NULL,
            f3_score FLOAT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS all_features_usage (
            id INTEGER NOT NULL,
            M_type TEXT NOT NULL,
            abs_S_Smin BOOLEAN NOT NULL,
            rel_S_Smin_semi_width BOOLEAN NOT NULL,
            rel_S_Smin_full_width BOOLEAN NOT NULL,
            abs_S_Smax BOOLEAN NOT NULL,
            rel_S_Smax_semi_width BOOLEAN NOT NULL,
            rel_S_Smax_full_width BOOLEAN NOT NULL,
            count_anomalies_S BOOLEAN NOT NULL,
            ratio_anomalies_S BOOLEAN NOT NULL,
            max_variation_S BOOLEAN NOT NULL,
            abs_T_Tmin BOOLEAN NOT NULL,
            rel_T_Tmin_semi_width BOOLEAN NOT NULL,
            rel_T_Tmin_full_width BOOLEAN NOT NULL,
            abs_T_Tmax BOOLEAN NOT NULL,
            rel_T_Tmax_semi_width BOOLEAN NOT NULL,
            rel_T_Tmax_full_width BOOLEAN NOT NULL,
            count_anomalies_T BOOLEAN NOT NULL,
            ratio_anomalies_T BOOLEAN NOT NULL,
            max_variation_T BOOLEAN NOT NULL,
            mean_correlation BOOLEAN NOT NULL,
            nb_measurements BOOLEAN NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dtc_parameters (
            id INTEGER NOT NULL,
            M_type TEXT NOT NULL,
            criterion TEXT NOT NULL,
            max_depth INT NOT NULL,
            min_samples_split INT NOT NULL,
            min_samples_leaf INT NOT NULL,
            max_features INT NOT NULL,
            max_leaf_nodes INT NOT NULL,       
            min_impurity_decrease FLOAT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rfc_parameters (
            id INTEGER NOT NULL,
            M_type TEXT NOT NULL,
            number_of_trees INTEGER NOT NULL,
            criterion TEXT NOT NULL,
            max_depth INT NOT NULL,
            min_samples_split INT NOT NULL,
            min_samples_leaf INT NOT NULL,
            max_features INT NOT NULL,
            max_leaf_nodes INT NOT NULL,       
            min_impurity_decrease FLOAT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mlp_parameters (
            id INTEGER NOT NULL,
            M_type TEXT NOT NULL,
            growth_rate INTEGER NOT NULL,
            dropout FLOAT NOT NULL,
            learning_rate FLOAT NOT NULL,
            epochs INT NOT NULL,
            factor FLOAT NOT NULL,
            patience INT NOT NULL,
            epsilon INT NOT NULL       
        )
    """)
    conn.commit()

if __name__ == "__main__" :
    main()