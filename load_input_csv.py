# Original Source: https://github.com/IRC-SPHERE/sphere-challenge/blob/master/jupyter/sphere-baseline.ipynb
import numpy as np
import os
from visualise_data import Sequence
import pandas as pd


public_data_path = 'data'
metadata_path = 'data/metadata'

# feature_functions = [np.mean,np.std]
# feature_names = ['mean','std']

feature_functions = [np.mean, np.std, np.min, np.median, np.max, np.sum]
feature_names = ['mean', 'std', 'min', 'median', 'max', 'sum']

number_feature_functions = len(feature_functions)


# All features
feature_source_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']

def load_input(feature,csv_name):

    if feature==0:
        feature_source_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']
    elif feature == 1:
        feature_source_names = ['acceleration', 'rssi', 'pir']
    elif feature == 2:
        feature_source_names = ['acceleration', 'rssi', 'video_living_room', 'video_kitchen', 'video_hallway']
    elif feature == 3:
        feature_source_names = ['pir', 'video_living_room', 'video_kitchen', 'video_hallway']
    elif feature == 4:
        feature_source_names = ['acceleration', 'rssi']
    elif feature == 5:
        feature_source_names = ['pir']
    elif feature == 6:
        feature_source_names = ['video_living_room', 'video_kitchen', 'video_hallway']

    column_names = []

    
    for train_test in ('train',):
        if train_test == 'train':
            print("Extracting features from training data.\n")
        else:
            print("Extracting features from test data.\n")

        # file number = number of current file e.g. 0,1,2, etc
        # file id = id number of current file e.g. 00005, 00004, 00008 etc
        # stub name = file number with 4 zeros before e.g. 00001, 00002, 00003 etc. 
        for file_number,file_id in enumerate(os.listdir('{}/{}/'.format(public_data_path, train_test))):
            #print("file_number: ",file_number)
            print("file_id: ",file_id)

            #stub_name = str(file_id).zfill(5)
            #print("Stub name: ",stub_name)

            # Extract features for all training data OR some subset of test data
            if train_test == 'train' or np.mod(file_number, 50) == 0:
                print ("Starting feature extraction for {}/{}".format(train_test, file_id))

            # Loads column names from each .csv file specified in metadata folder
            data = Sequence(metadata_path, '{}/{}/{}'.format(public_data_path, train_test, file_id))
            data.load()

            # Populate column name list
            # Only needs to be done on first iteration
            # Because column names will be the same between datasets 
            # result is 366 column names (6*num(original features)) ... we generate 6* because we have 6 feature functions: (mean, max, min, std, median, sum)
            if(len(column_names)==0):
                if feature==0:
                    feature_load = data.iterate_all()
                elif feature == 1:
                    feature_load = data.iterate_no_video()
                elif feature == 2:
                    feature_load = data.iterate_no_pir()
                elif feature == 3:
                    feature_load = data.iterate_no_acceleration()
                elif feature == 4:
                    feature_load = data.iterate_acceleration_only()
                elif feature == 5:
                    feature_load = data.iterate_pir_only()
                elif feature == 6:
                    feature_load = data.iterate_video_only()
                
                #for lu, modalities in data.iterate_all():
                for lu,modalities in feature_load:
                    print("lu: ",lu)
                    print("modalities: ",modalities)
                    if feature==5:
                        for modality, modality_name in zip(modalities, feature_source_names):
                            print("Modality Name: ",modality_name)
                            if(feature==5):
                                for column_name, column_data in modalities.transpose().iterrows():
                                    for feature_name in feature_names:
                                        column_names.append('{0}_{1}_{2}'.format(modality_name, modality, feature_name))
                            else:
                                for column_name, column_data in modality.transpose().iterrows():
                                    for feature_name in feature_names:
                                        column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))

                    else:
                        for modality, modality_name in zip(modalities, feature_source_names):
                            print("Modality Name: ",modality_name)
                            if(feature==5):
                                #pir_data = data.pir
                                for column_name, column_data in data.pir.transpose().iterrows():
                                    for feature_name in feature_names:
                                        # string = str(modality_name)+str(column_name)+str(feature_name)
                                        column_names.append('{0}_{1}_{2}'.format(modality_name, modality, feature_name))
                                        # column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))
                            else:
                                for column_name, column_data in modality.transpose().iterrows():
                                    for feature_name in feature_names:
                                        column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))

                    # Break here 
                    break 

            """
            Here, we will extract some features from the data. We will use the Sequence.iterate function. 

            This function partitions the data into one-second dataframes for the full data sequence. The first element 
            (which we call lu) represents the lower and upper times of the sliding window, and the second is a list of
            dataframes, one for each modality. The dataframes may be empty (due to missing data), and feature extraction 
            has to be able to cope with this! 

            The list rows will store the features extracted for this dataset
            """
            # Gets input features for each 1 second interval
            # If there is no data for this time period in other sensors, the column value is assigned a NaN value
            # This is why we must use the mean, max, min etc as we are using a time period rather than each value
            # We must group the timings into some sort of interval due to the disparity in timings across all of our sensors.
            rows = []
            #for ri, (lu, modalities) in enumerate(feature_load):
            if feature==0:
                feature_load = enumerate(data.iterate_all())
            elif feature == 1:
                feature_load = enumerate(data.iterate_no_video())
            elif feature == 2:
                feature_load = enumerate(data.iterate_no_pir())
            elif feature == 3:
                feature_load = enumerate(data.iterate_no_acceleration())
            elif feature == 4:
                feature_load = enumerate(data.iterate_acceleration_only())
            elif feature == 5:
                feature_load = enumerate(data.iterate_pir_only())
            elif feature == 6:
                feature_load = enumerate(data.iterate_video_only())

            for ri, (lu,modalities) in feature_load: 
                row = []

                """
                Iterate over the sensing modalities. The order is given in modality_names. 

                Note: If you want to treat each modality differently, you can do the following: 

                for ri, (lu, (accel, rssi, pir, vid_lr, vid_k, vid_h)) in enumerate(data.iterate()):
                   row.extend(extract_accel(accel))
                   row.extend(extract_rssi(rssi))
                   row.extend(extract_pir(pir))
                   row.extend(extract_video(vid_lr, vid_k, vid_h))

                """
                if feature==5:
                    for name, column_data in modalities.transpose().iterrows():
                        if len(column_data) > 3:
                                """
                                Extract the features stored in feature_functions on the column data if there is sufficient 
                                data in the dataframe. 
                                """
                                # print("Modlaity: ",modality)
                                # print("Column Name: ",name)
                                # print("Column data: ",column_data)
                                row.extend(map(lambda ff: ff(column_data), feature_functions))
                        else:
                            """
                            If no data is available, put nan placeholders to keep the column widths consistent
                            """
                            row.extend([np.nan] * number_feature_functions)
                else:
                    for modality in modalities:
                        """
                        The accelerometer dataframe, for example, has three columns: x, y, and z. We want to extract features 
                        from all of these, and so we iterate over the columns here. 
                        """
                        for name, column_data in modality.transpose().iterrows():
                        #for name, column_data in data.pir.transpose().iterrows():
                            #for entry in column_data:
                                if len(column_data) > 3:
                                    """
                                    Extract the features stored in feature_functions on the column data if there is sufficient 
                                    data in the dataframe. 
                                    """
                                    # print("Modlaity: ",modality)
                                    # print("Column Name: ",name)
                                    # print("Column data: ",column_data)
                                    row.extend(map(lambda ff: ff(column_data), feature_functions))
                                else:
                                    """
                                    If no data is available, put nan placeholders to keep the column widths consistent
                                    """
                                    row.extend([np.nan] * number_feature_functions)
                # Do a quick sanity check to ensure that the feature names and number of extracted features match
                assert len(row) == len(column_names)

                 # Append the row to the full set of features
                rows.append(row)

                # Report progress 
                if train_test is 'train':
                    if np.mod(ri + 1, 50) == 0:
                        print ("{:5}".format(str(ri + 1))),

                    if np.mod(ri + 1, 500) == 0:
                        print
                """
            At this stage we have extracted a bunch of simple features from the data. In real implementation, 
            it would be advisable to look at more interesting features, eg

              * acceleration: link
              * environmental: link
              * video: link

            We will save these features to a new file called 'columns.csv' for use later. This file will be located 
            in the name of the training sequence. 
            """
            df = pd.DataFrame(rows)
            df.columns = column_names
            df.to_csv('{}/{}/{}/columns_1000ms_{}.csv'.format(public_data_path, train_test, file_id, csv_name), index=False)

            if train_test is 'train' or np.mod(file_number, 50) == 0:
                if train_test is 'train': print 
                print ("Finished feature extraction for {}/{}\n".format(train_test, file_id))


# feature == 0 => All csv input files
load_input(feature=0,csv_name="all_inputs")

# feature == 1 => No video_*.csv
load_input(feature=1,csv_name="no_video")

# feature == 2 => No PIR.csv
load_input(feature=2,csv_name="no_pir")

# feature == 3 => No acceleration.csv
load_input(feature=3,csv_name="no_acceleration")

# feature == 4 => acceleration.csv only
load_input(feature=4,csv_name="acceleration_only")

# feature == 5 =>  PIR.csv only
load_input(feature=5,csv_name="pir_only")

# feature == 6 => video_*.csv only
load_input(feature=6,csv_name="video_only")





