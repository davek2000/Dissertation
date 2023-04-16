import pandas as pd
import numpy as np
import sklearn
import json 
from itertools import cycle
import seaborn as sns 
import matplotlib.pyplot as plt
import os
from sklearn.multioutput import MultiOutputRegressor
import time

def load_sequence(file_id,feature_input_file_name):
    filename = str(file_id).zfill(5)

    df = pd.read_csv('{}/train/{}/{}.csv'.format(public_data_path, filename,feature_input_file_name))
    data = df.values
    target = np.asarray(pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, filename)))[:, 2:]

    return data, target


def load_sequences(file_ids,feature_file_name):
    x_es = []
    y_es = []

    for file_id in file_ids:
        data, target = load_sequence(file_id,feature_file_name)

        x_es.append(data)
        y_es.append(target)

    return np.row_stack(x_es), np.row_stack(y_es)

public_data_path = 'data'

feature_input_file_name = "columns_1000ms_all_inputs"
# feature_input_file_name = "columns_1000ms_no_video"
# feature_input_file_name = "columns_1000ms_no_pir"
# feature_input_file_name = "columns_1000ms_no_acceleration"
# feature_input_file_name = "columns_1000ms_acceleration_only"
# feature_input_file_name = "columns_1000ms_pir_only"
# feature_input_file_name = "columns_1000ms_video_only"
# Load the training and testing data 
train_x, train_y = load_sequences([1, 2, 3, 4, 5, 6, 7, 8],feature_input_file_name)
test_x, test_y = load_sequences([9, 10],feature_input_file_name)

print ("Check whether the train/test features are all finite (before imputation)")
print ('All training data finite:', np.all(np.isfinite(train_x)))
print ('All testing data finite:', np.all(np.isfinite(test_x)))
print 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(train_x)
train_x = imputer.transform(train_x)
test_x = imputer.transform(test_x)

print ("Check whether the train/test features are all finite (after imputation)")
print ('All training data finite:', np.all(np.isfinite(train_x)))
print ('All testing data finite:', np.all(np.isfinite(test_x)))
print 

# Load the label names 
metadata_path = 'data/metadata'
labels = json.load(open(metadata_path + '/annotations.json'))
n_classes = len(labels)

"""
Note, not all data is annotated, so we select only the annotated rows
"""
train_y_has_annotation = np.isfinite(train_y.sum(1))
train_x = train_x[train_y_has_annotation]
train_y = train_y[train_y_has_annotation]

test_y_has_annotation = np.isfinite(test_y.sum(1))
test_x = test_x[test_y_has_annotation]
test_y = test_y[test_y_has_annotation]

from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
train_x = st_x.fit_transform(train_x)
test_x = st_x.transform(test_x)


"""
Print simple statistics regarding the number of instances
"""
print ("Training data shapes:")
print ("train_x.shape: {}".format(train_x.shape))
print ("train_y.shape: {}".format(train_y.shape))
print 

print ("Testing data shapes")
print ("test_x.shape: {}".format(test_x.shape))
print ("test_y.shape: {}".format(test_y.shape))
print

activity_names = json.load(open(metadata_path + '/annotations.json', 'r'))
class_weights = np.asarray(json.load(open(metadata_path + '/class_weights.json', 'r')))

class_prior = train_y.mean(0)

df = pd.DataFrame({
        'Activity': activity_names, 
        'Class Weight': class_weights,
        'Prior Class Distribution': class_prior
    })

# df.set_index('Activity', inplace=True)
# # reset colour palette
# current_palette = cycle(sns.color_palette())
# df.plot(
#     kind='bar',
#     width=1.0,
#     subplots=True,
#     color=[next(current_palette), next(current_palette)], 
#     edgecolor='white',
#     linewidth=1
# )

#plt.show()

se_cols = ['start', 'end']

num_lines = 0




def brier_score(given, predicted, weight_vector): 
        # a = given-predicted
        # b = np.power(a,2.0)
        # c = b.dot(weight_vector)
        # print(c)
        return np.power(given - predicted, 2.0).dot(weight_vector).mean()

def baseline():
     # For comparison to the KNN model, test the prior class distribution  
    prior_matrix = np.repeat(class_prior.reshape((1, -1)), test_y.shape[0], axis=0)
    # convert_predictions_to_probabilities(prior_matrix)
    prior_brier_score = brier_score(test_y, prior_matrix, class_weights)

    print ("Brier score on test using the prior class probability as a constant model")
    print (prior_brier_score)
    print

def knn_model(k_value,p_value):
    from sklearn.neighbors import NearestNeighbors
    """
    Define a simple class that inherits from sklearn.neighbors.NearestNeighbors. 
    We will adjust the fit/predict as necessary
    """
    class ProbabilisticKNN(NearestNeighbors): 
        def __init__(self,n_neighbours,p_value): 
            super(ProbabilisticKNN, self).__init__(n_neighbors=n_neighbours,n_jobs=-1,p=p_value)

            self.train_y = None

        def fit(self, train_x, train_y): 
            """
            The fit function requires both train_x and train_y. 
            See 'The selected model' section above for explanation
            """

            self.train_y = np.copy(train_y)

            super(ProbabilisticKNN, self).fit(train_x)

        def predict_proba(self, test_x): 
            """
            This function finds the k closest instances to the unseen test data, and 
            averages the train_labels of the closest instances. 
            """

            # Find the nearest neighbours for the test set
            test_neighbours = self.kneighbors(test_x, return_distance=False)

            # Average the labels of these for prediction
            return np.asarray(
                [self.train_y[inds].mean(0) for inds in test_neighbours]
            )

    # Learn the KNN model 
    knn = ProbabilisticKNN(n_neighbours=k_value,p_value=p_value)
    knn.fit(train_x, train_y)

    # Predict on the test instances
    test_predicted = knn.predict_proba(test_x)

    greater_one_count =0
    for row in test_predicted:
        if np.sum(row) > 1.01:
            #print("Prediction sums greater than 1")
            greater_one_count = greater_one_count+1
            print(row)
            print("SUM: ",np.sum(row))
                
    print("Greater One Count: ",greater_one_count)
    knn_brier_score = brier_score(test_y, test_predicted, class_weights)

    print ("Brier score on test set with the KNN model")
    print (knn_brier_score)
    return knn_brier_score 

def linear_regression():
    from sklearn.linear_model import LinearRegression

    reg = MultiOutputRegressor(LinearRegression(n_jobs=-1))
    reg.fit(train_x,train_y)

    y_pred = reg.predict(test_x)
    #convert_predictions_to_probabilities(y_pred)

    # greater_one_count =0
    # for row in y_pred:
    #     if np.sum(row) > 1.01:
    #         #print("Prediction sums greater than 1")
    #         greater_one_count = greater_one_count+1
    #         print(row)
    #         print("SUM: ",np.sum(row))
                
    # print("Greater One Count: ",greater_one_count)

    y_pred_original = reg.predict(test_x)

    for i in range(y_pred.shape[0]):
        sum = np.sum(y_pred[i])
        for j in range(y_pred.shape[1]):
            entry = y_pred[i][j]
            y_pred[i][j] = entry/sum

    # LR sums do add to 1
    # Hence this code is un-necessary
    # for row in y_pred:
    #     sum = np.sum(row)
    #     if sum < 0.99 or sum > 1.01:
    #         print("Less than 0.99 sum OR > 1.01")

    print ("Brier score on test using reg model (Before PROB conversion)")

    reg_briar = brier_score(test_y,y_pred_original,class_weights)
    print(reg_briar)
    
    print ("Brier score on test using reg model (After PROB conversion)")
    reg_briar = brier_score(test_y,y_pred,class_weights)
    print(reg_briar)

def random_forest(n_estimators,max_depth,max_features):
    from sklearn.ensemble import RandomForestRegressor
    reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, 
                                                     max_depth=max_depth, 
                                                     max_features=max_features,
                                                     n_jobs=-1))
    reg.fit(train_x,train_y)

    y_pred = reg.predict(test_x)
    # reg_briar = brier_score(test_y,y_pred,class_weights)
    # print(reg_briar)


    # print(len(reg.estimators_))

    # for estimator in reg.estimators_:
    #     feature_importances = estimator.feature_importances_
        # print(estimator.feature_importances_)

    # n_estimators = 10
    # 30.19 seconds with 1 processor
    # 12.75 seconds with all processor's

    # RF sums do not add to 1
    # Hence we need to convert rows to probabilities (unlike LR)

    # for row in y_pred:
    #     sum = np.sum(row)
    #     if sum < 0.99 or sum > 1.01:
    #         print("Less than 0.99 sum OR > 1.01")

    y_pred_original = y_pred.copy()

    # for i in range(y_pred.shape[0]):
    #     sum = np.sum(y_pred[i])
    #     for j in range(y_pred.shape[1]):
    #         entry = y_pred[i][j]
    #         y_pred[i][j] = entry/sum

    # print ("Brier score on test using random forest model (Before PROB conversion)")
    # reg_briar = brier_score(test_y,y_pred_original,class_weights)
    # print(reg_briar)

    print ("Brier score on test using random forest model (After PROB conversion)")
    reg_briar = brier_score(test_y,y_pred,class_weights)
    print(reg_briar)
    return reg_briar

def neural_network(learning_rate,batch_size,epochs,dropout_rate):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    model = Sequential()
    model.add(Dense(80, activation='relu',input_dim=train_x.shape[1]))
    model.add(Dropout(dropout_rate))
    #model.add(Dense(180, activation='relu'))
    #model.add(Dense(180, input_dim=366, activation='relu'))
    #model.add(Dense(366, activation='relu'))
    model.add(Dense(20, activation='softmax'))  # Softmax for probability distribution

    from keras.optimizers import Adam
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate), 
                  metrics=['accuracy'],
                  loss_weights=class_weights)


    #model.fit(train_x,train_y,epochs=20,batch_size=16,verbose=0)
    model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size,verbose=0)
    # print("Evaluate on test data")
    # results = model.evaluate(test_x, test_y, batch_size=128)
    # print("test loss, test acc:", results)
    print(model.summary())
    y_pred = model.predict([test_x])
    #print(y_pred)

    print ("Brier score on test using neural network model (Before PROB conversion)")
    reg_briar = brier_score(test_y,y_pred,class_weights)
    print(reg_briar)

    for i in range(y_pred.shape[0]):
        sum = np.sum(y_pred[i])
        for j in range(y_pred.shape[1]):
            entry = y_pred[i][j]
            y_pred[i][j] = entry/sum

    print ("Brier score on test using random forest model (After PROB conversion)")
    reg_briar = brier_score(test_y,y_pred,class_weights)
    print(reg_briar)
    return reg_briar

def test_baseline():
    ###########################################################################################################
    # Baseline model
    baseline()
    ###########################################################################################################

def test_kNN():
    ###########################################################################################################
    # kNN model
    print("kNN Model:")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    knn_brier_score = []

    # k_range = np.power(2, range(8))
    # # Hyper-param: k
    # for k in k_range:
    #     print("k value: ",k)
    #     knn_brier_score.append(knn_model(k_value=k))

    # # Hyper-param: p
    # k_range = ['1','2']
    # knn_brier_score.append(knn_model(k_value=32,p_value=1))
    # knn_brier_score.append(knn_model(k_value=32,p_value=2))

    # # Plot hyper-param
    # plt.plot(k_range,knn_brier_score,label='Brier score WRT p')
    # plt.xticks(k_range)
    # plt.xlabel("p Value"); plt.ylabel("Brier Score")
    # plt.title("kNN Hyper-parameter tuning: p")
    # plt.show()

    knn_model(k_value=32,p_value=1)
    ###########################################################################################################

def test_linear_regression():
    ##########################################################################################################
    # Linear Regression Model
    print("Linear Regression Model:")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    linear_regression()
    ##########################################################################################################

def test_random_forest():
    ###########################################################################################################
    # Random Forest Model
    # print("Random Forest Model:")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # brier_score_list = []
    # rf_timings = []

    # # Hyper-parameter: n_estimators
    # n_estimators = [1,5,10,20,30,40,50,60,70]
    # for n_e in n_estimators:
    #     start = time.time()  
    #     brier_score_list.append(random_forest(n_estimators=n_e,max_depth=None,max_features=1.0))
    #     stop = time.time()
    #     rf_timings.append((stop-start))

    # plt.plot(n_estimators,rf_timings,label='Execution time WRT n_estimators')
    # plt.plot(n_estimators,brier_score_list)
    # plt.xticks(n_estimators); #plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
    # plt.xlabel("n_estimators"); plt.ylabel("Time (seconds)")
    # plt.title("Random Forest Hyper-parameter tuning: n_estimators")
    # plt.show()

    # # Hyper-parameter: max_depth
    # max_depth_values = [10,20,30,40,50]
    # brier_score_list = []
    # rf_timings = []
    # # for depth_value in max_depth_values:
    # #     start = time.time()
    # #     brier_score_list.append(random_forest(n_estimators=10,max_depth=depth_value,max_features=1.0))
    # #     stop = time.time()
    # #     rf_timings.append((stop-start))

    # #plt.plot(max_depth_values,rf_timings,label='Brier Score WRT max_depth')
    # plt.plot(max_depth_values,brier_score_list)
    # plt.xticks(max_depth_values); #plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
    # plt.xlabel("max_depth"); plt.ylabel("Time (seconds)")
    # plt.title("Random Forest Hyper-parameter tuning: max_depth")
    # plt.show()

    # # Hyper-parameter: max_features
    # max_features = ["sqrt", "log2", None]
    # brier_score_list = []
    # rf_timings = []

    # for max_feature in max_features:
    #     start = time.time()
    #     brier_score_list.append(random_forest(n_estimators=10,max_depth=10,max_features=max_feature))
    #     stop = time.time()
    #     rf_timings.append((stop-start))

    # #plt.plot(max_depth_values,rf_timings,label='Brier Score WRT max_depth')
    # plt.plot([0,1,2],brier_score_list)
    # plt.xticks(range(len(max_features)),['sqrt','log2','None'] ); #plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
    # plt.xlabel("max_features"); plt.ylabel("Brier Score")
    # plt.title("Random Forest Hyper-parameter tuning: max_features")
    # plt.show()

    random_forest(n_estimators=10,max_depth=10,max_features="log2")
    ###########################################################################################################

def test_neural_network():
    ###########################################################################################################
    # # Neural Network Model
    print("Neural Network Model:")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # # Hyper-parameter: learning_rate 
    # learning_rates = [0.01,0.001,0.0001,0.00001]
    # brier_score_list = []
    # nn_timing=  []
    # for lr in learning_rates:
    #     start = time.time()
    #     brier_score_list.append(neural_network(learning_rate=lr))
    #     stop = time.time()
    #     nn_timing.append((stop-start))

    # # #plt.plot(max_depth_values,rf_timings,label='Brier Score WRT max_depth')
    # plt.plot(range(len(learning_rates)),nn_timing)
    # #plt.xticks(range(len(max_features)),['sqrt','log2','None'] ); #plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
    # plt.xticks(range(len(learning_rates)),["0.01","0.001","0.0001","0.00001"])
    # plt.xlabel("learning_rate"); plt.ylabel("Time (seconds)")
    # plt.title("Neural Network Hyper-parameter tuning: learning_rate")
    # plt.show()

    # Hyper-parameter: batch_size

    # brier_score_list = []
    # nn_timing=  []

    # batch_sizes = [16,32,64,128]
    # for batch in batch_sizes:
    #     start = time.time()
    #     brier_score_list.append(neural_network(learning_rate=0.001,batch_size=batch))
    #     stop = time.time()
    #     nn_timing.append((stop-start))

    # plt.plot(range(len(batch_sizes)),brier_score_list)
    # #plt.xticks(range(len(max_features)),['sqrt','log2','None'] ); #plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
    # plt.xticks(range(len(batch_sizes)),batch_sizes)
    # plt.yticks([0.3,0.25,0.2])
    # plt.xlabel("batch_size"); plt.ylabel("Brier Score")
    # plt.title("Neural Network Hyper-parameter tuning: batch_size")
    # plt.show()

    # Hyper-paramater: epoch_size
    # brier_score_list = []
    # nn_timing=  []

    # epoch_sizes = [1,5,10,20,30,40,50]
    # for epoch in epoch_sizes:
    #     start = time.time()
    #     brier_score_list.append(neural_network(learning_rate=0.001,batch_size=64, epochs=epoch))
    #     stop = time.time()
    #     nn_timing.append((stop-start))

    # plt.plot(range(len(epoch_sizes)),nn_timing)
    # #plt.xticks(range(len(max_features)),['sqrt','log2','None'] ); #plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
    # plt.xticks(range(len(epoch_sizes)),epoch_sizes)
    # #plt.yticks([0.3,0.25,0.2])
    # plt.xlabel("epoch_value"); plt.ylabel("Time (seconds)")
    # plt.title("Neural Network Hyper-parameter tuning: epoch_value")
    # plt.show()

    # Hyper-parameter: Dropout-rate
    # brier_score_list = []
    # nn_timing=  []

    # dropout_rate = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    # for dropout in dropout_rate:
    #     start = time.time()
    #     brier_score_list.append(neural_network(learning_rate=0.001,batch_size=64, epochs=20, dropout_rate=dropout))
    #     stop = time.time()
    #     nn_timing.append((stop-start))

    # plt.plot(range(len(dropout_rate)),nn_timing)
    # #plt.xticks(range(len(max_features)),['sqrt','log2','None'] ); #plt.yticks([0.5,0.4,0.3,0.2,0.1,0])
    # plt.xticks(range(len(dropout_rate)),dropout_rate)
    # #plt.yticks([0.3,0.25,0.2])
    # plt.xlabel("dropout_rate"); plt.ylabel("Time (seconds)")
    # plt.title("Neural Network Hyper-parameter tuning: dropout_rate")
    # plt.show()

    neural_network(learning_rate=0.001,batch_size=64,epochs=20,dropout_rate=0.4)
    ###########################################################################################################

#test_neural_network()
# given=[0.2,0.5,0.2]
# given = np.array([0.2,0.5,0.2])
# # predicted = [0.1,0.5,0.1]
# predicted = np.array([0.1,0.5,0.1])
# # weights = [1,1,1]
# weights = [1,1,1]
# print(brier_score(given,predicted,weights))
# baseline()

# print("Baseline:")
# start = time.time()
# test_baseline()
# stop = time.time()
# print("Duration: ",(stop-start))

# print("Linear Regression")
# start = time.time()
# test_linear_regression()
# stop = time.time()
# print("Duration: ",(stop-start))

# print("kNN")
# start = time.time()
# test_kNN()
# stop = time.time()
# print("Duration: ",(stop-start))

# print("Random Forest")
# start = time.time()
# test_random_forest()
# stop = time.time()
# print("Duration: ",(stop-start))

# print("Neural Network")
# start = time.time()
# test_neural_network()
# stop = time.time()
# print("Duration: ",(stop-start))

def plot_results():
    baseline_score = [0.2930, 0.2930, 0.2930, 0.2930, 0.2930, 0.2930, 0.2930]
    lr_score = [0.2824, 0.2725, 0.2872, 0.2858, 0.2759, 0.2878, 0.2898]
    knn_score = [0.2538, 0.2707, 0.2549, 0.2776, 0.2706, 0.3274, 0.2794]
    rf_score = [0.2565, 0.2660, 0.26, 0.267, 0.2693, 0.2878, 0.2719]
    nn_score = [0.2367, 0.2555, 0.2374, 0.2617, 0.2567, 0.2875, 0.2609]

    baseline_time = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    lr_time = [5.62, 1.2748,4.28,4.41,0.57,0.81,2.99]
    knn_time = [3.45, 1.083,5.61,3.06,0.64,0.98,2.50]
    rf_time = [5.6, 5.0526,5.60,5.33,5.36,4.05,5.15]
    nn_time = [6.63, 5.786,5.93,6.34,5.60,5.82,5.81]

    feature_list = ["All inputs", "No video", "No PIR", "No acceleration", "Acceleration only", "PIR only", "Video only"]

    # plt.plot(range(len(feature_list)),baseline_score)
    # plt.plot(range(len(feature_list)),lr_score)
    # plt.plot(range(len(feature_list)),knn_score)
    # plt.plot(range(len(feature_list)),rf_score)
    # plt.plot(range(len(feature_list)),nn_score)

    # plt.plot(range(len(feature_list)),baseline_time)
    # plt.plot(range(len(feature_list)),lr_time)
    # plt.plot(range(len(feature_list)),knn_time)
    # plt.plot(range(len(feature_list)),rf_time)
    # plt.plot(range(len(feature_list)),nn_time)

    plt.bar(0,baseline_time[0])
    plt.bar(1,lr_time[0])
    plt.bar(2,knn_time[0])
    plt.bar(3,rf_time[0])
    plt.bar(4,nn_time[0])

    #plt.xticks(range(len(feature_list)),feature_list)
    plt.xticks([0,1,2,3,4],["Baseline","Linear Regression", "kNN","Random Forest","Neural Network"])
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of model performance using all available input features")
    #plt.legend(["Baseline", "Linear Regression", "kNN", "Random Forest", "Neural Network"])


    plt.show()

plot_results()

