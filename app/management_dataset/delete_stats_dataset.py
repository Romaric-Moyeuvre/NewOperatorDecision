## IMPORT ##########

import os

####################

def delete_measures_dataset() :

    MAIN_PATH = os.getenv("MINMAX_ODM_MAINPATH").replace("\r", "").replace("\n", "")

    alarm = ["/temperature/","/salinity/"]
    label = ["false_alarm","true_alarm"]

    for alarm_text in alarm :
        for label_text in label :
            delete_path = MAIN_PATH+"/data/dataset_custom/dataset_stats/"+alarm_text+label_text
            for file in os.listdir(delete_path) :
                if file.endswith(".npy") == False :
                    continue
                os.remove(delete_path+"/"+file)

if __name__ == "__main__" :
    delete_measures_dataset()