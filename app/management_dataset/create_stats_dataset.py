## IMPORT ##########

import os
import numpy
import netCDF4

from analysis import *
from preprocessing import *

from clear_cache import clear as clear_cache

####################

def create_stats_dataset() :

    MAIN_PATH = os.getenv("MINMAX_ODM_MAINPATH").replace("\r", "").replace("\n", "")
    alarm_ = {"b'T'": 0, "b'S'": 1}
    alarm_text = {0:"temperature/", 1:"salinity/"}
    label_text = {0:"false_alarm/", 1:"true_alarm/"}

    dataset_path = MAIN_PATH+"/data/dataset_raw/dataset_v4.4"
    save_path = MAIN_PATH+"/data/dataset_custom/dataset_stats/"

    for file in os.listdir(dataset_path) :

        if file.endswith(".nc") == False :
            continue

        data = netCDF4.Dataset(dataset_path+"/"+file, 'r', format="NETCDF4")
        number_of_profiles = (data["PARAM"].shape)[0]
        for index in range(number_of_profiles) :

            alarm = alarm_[str(data["PARAM"][index])]
            label = float(data["FALSEorTRUE"][index])

            pressure = numpy.array(data["PRES"][index][:])
            psal = numpy.array(data["PSAL"][index][:])
            psal_min = numpy.array(data["PSAL_MIN"][index][:])
            psal_med = numpy.array(data["PSAL_MED"][index][:])
            psal_max = numpy.array(data["PSAL_MAX"][index][:])
            temp = numpy.array(data["TEMP"][index][:])
            temp_min = numpy.array(data["TEMP_MIN"][index][:])
            temp_med = numpy.array(data["TEMP_MED"][index][:])
            temp_max = numpy.array(data["TEMP_MAX"][index][:])

            dataset_FULL = [pressure, psal, psal_min, psal_med, psal_max, temp, temp_min, temp_med, temp_max]
            dataset_FULL = filter(dataset_FULL)

            if len(dataset_FULL[0]) > 0 :
                dataset_FULL = sort(dataset_FULL)
                dataset_STATS = get_stats(dataset_FULL)
                filename = file[:-3]+'_'+str(index)+'.npy'
                numpy.save(save_path+alarm_text[alarm]+label_text[label]+filename, dataset_STATS)

if __name__ == "__main__" :
    create_stats_dataset()