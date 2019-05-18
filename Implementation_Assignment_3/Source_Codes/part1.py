#!/usr/bin/env python
# coding: utf-8

#part_1_final
#copy 2 root works all good now make the tree work
#main the get split works for the root copy1
import numpy as np
np.set_printoptions(threshold=np.nan)
import csv

import matplotlib.pyplot as plt
import time


#create an array to store the positions in the sorted subtable where the values
start_time = time.time()
indices_where_change = np.zeros([])
training_accuracy_list =[]
testing_accuracy_list = []
max_depth_list=[]

plt.xlabel("Depth of the tree")
plt.ylabel("Accuracy Percentage")
plt.title("Decision Tree Accuracy vs Depth of the Tree")

def open_csv(filename):
    f = open(filename + ".csv", 'r')
    reader = csv.reader(f)
    labels = []
    features = []
 
    for i, row in enumerate(reader):
        labels.append(float(row[0]))
        features.append([float(x) for x in row[1:]])
    features = np.array(features)
    
    labels = np.array(labels)
#     features = features [0:10,...]
#     labels = labels[0:10,...]
    labels[labels == 3.0] = 1
    labels[labels == 5.0] = -1
 
    return labels, features

#maintain an array for Benefit values for all features
benefit_array_for_one_feature = np.zeros(101)
#maintain an array for indices for the threshold values that give the maximum benefit for all features
max_benefit_indices_for_every_feature = np.zeros(101)


def gini_data(y): #####################MAKE THIS EFFICEINT
    #count the Y equal to one
    c_root_pos=np.count_nonzero(y==1)
    c_root_total = len(y)
#     print("c_root_positive "+str(c_root_pos))
#     print("c root total "+str(c_root_total))
    return ( 1- (c_root_pos/c_root_total)**2 -((c_root_total-c_root_pos)/c_root_total)**2 )


def test_split(index, value, dataset):
    left, right = list(), list()
    #print("index of feature is "+str(index))
    for row in dataset:
        if row[index] < value:
            #print(" left \n"+str(row[index]))
            left.append(row)
        else:
            #print(" right \n"+str(row[index]))
            right.append(row)
    #rint(str(len(left))+" classified as left by test split")
    #rint(str(len(right))+"classified as right by test split")
    return left, right


def get_split(dataset):
    nump = []
    for row in dataset:
        nump.append(row)
    #print(str(len(dataset)) + "values in the dataset in get split")
    dataset = np.array(nump)
    Y = dataset[..., 0]
    X = dataset[..., 1:]

    for col in range(X.shape[1]):

        single_feature = X[..., col]
        # print("size of single feature "+(str(single_feature.shape)+"\n\n"))

        # stack y_to_take and X[col]
        stacked = np.column_stack((Y, single_feature))
        # pass as arg the indices of the sorted 2nd column
        sorted = stacked[np.argsort(stacked[:, 1])]

        y_to_take = sorted[..., 0]

        # gives the positions of the x values where the values change
        indices_where_change = np.where(y_to_take[:-1] != y_to_take[1:])[0]
        # print(str(indices_where_change))

        # just a redundant thing
        indices_where_Actual_thresholds_are = indices_where_change  # print("y to take \n"+str(y_to_take))
        # print("indices change see \n\n"+str(indices_where_Actual_thresholds_are))

        # lets iterate for each position within a feature  where there is a change in y lable after we have sorted x values
        for i in range(len(indices_where_Actual_thresholds_are)):
            # positive part
            CL_pos = np.count_nonzero(y_to_take[:indices_where_Actual_thresholds_are[i] + 1] == 1)
            # print("non zero count is "+str(CL_pos))
            CL_total = len(y_to_take[:indices_where_Actual_thresholds_are[i] + 1])
            # print("CL total"+str(CL_total))
            CL_neg = CL_total - CL_pos
            p_plus = CL_pos / (CL_total)
            p_neg = CL_neg / (CL_total)
            UAL = 1 - (p_plus) ** 2 - (p_neg) ** 2

            # print("\n\nUAL value is : "+str(UAL))

            # negative part
            CR_pos = np.count_nonzero(y_to_take[indices_where_Actual_thresholds_are[i] + 1:] == 1)
            CR_total = len(y_to_take[indices_where_Actual_thresholds_are[i] + 1:])
            CR_neg = CR_total - CR_pos
            p_plus_r = CR_pos / (CR_total)
            p_neg_r = CR_neg / (CR_total)
            UAR = 1 - (p_plus_r) ** 2 - (p_neg_r) ** 2

            # print("\n\nUAR value is : "+str(UAR))

            #             if(i % 100  == 99):
            #                 print("CL positive is " +str(CL_pos))
            #                 print("CL total"+str(CL_total))# = len(y_to_take[indices_where_Actual_thresholds_are[i] + 1:])
            #                 print("cl neg"+str(CL_neg))# = CR_total - CR_pos
            #                 print("p plus"+str(p_plus))#p_plus_r = CR_pos / (CR_total)
            #                 print("p neg "+str(p_neg))#p_neg_r = CR_neg / (CR_total)

            #                 print("CR positive is " +str(CR_pos))
            #                 print("CR total"+str(CR_total))# = len(y_to_take[indices_where_Actual_thresholds_are[i] + 1:])
            #                 print("cr neg"+str(CR_neg))# = CR_total - CR_pos
            #                 print("p plus r"+str(p_plus_r))#p_plus_r = CR_pos / (CR_total)
            #                 print("p neg r"+str(p_neg_r))#p_neg_r = CR_neg / (CR_total)

            #                 print("\n\nUAL value is : "+str(UAL))
            #                 print("\n\nUAR value is : "+str(UAR))

            pl = (CL_total) / len(y_to_take)
            # print("pl is "+str(pl))
            pr = (CR_total) / len(y_to_take)
            # print("pr is "+str(pr))

            # Benefit of splitting at this threshold i (length of this value varies among different features)
            Benefit_of_split_at_this_i = gini_data(y_to_take) - pl * UAL - pr * UAR

            # Store the benefit of splitting at this threhold if it exceeds the previously stored value for this feature
            if (Benefit_of_split_at_this_i > benefit_array_for_one_feature[col + 1]):
                # print("benefit grater at i, threshold pos "+str(Benefit_of_split_at_this_i)+ " "+str(i))
                benefit_array_for_one_feature[col + 1] = Benefit_of_split_at_this_i
                # also store the indices for these max benefit positions-- for each col value, we have a different value
                max_benefit_indices_for_every_feature[col + 1] = indices_where_Actual_thresholds_are[i]

    # after iterating over all the threshold values for all the features
    # find the index of the maximum benefit value from the array that stores maximum benefit for all the features
    # this index corresponds to the feature position that gave you the maximum benefit among all the features
    attribute_max_benefit = np.argmax(benefit_array_for_one_feature[1:])
    # print("position of argmax "+str(attribute_max_benefit))

    # now you got the position/ index of the feature corresponding to the actual maximum benefit value among all features

    #print("Calculating max benefit value ")
    #print("\nmax benefit val among all the maximum benefits from all features :  " + str(
    #    max(benefit_array_for_one_feature[1:])))

    # now we need to find the index of the threshold value within this feature column
    index_of_thres_with_max_benefit = max_benefit_indices_for_every_feature[attribute_max_benefit + 1]
    #print("index_of_thres_with_max_benefit" + str(index_of_thres_with_max_benefit))
    # NOTE: You wont see the threshold value at 3257 position of the original file though coz this position is after sorting

    # now, we need to find the actual value of that threshold since we only have the index of the
    # threshold within the feature column
    # however, note that that index that you got is for the sorted feature column, you wasted 3 days because
    # you did not take care of this fact

    # lets get the column of that feature and sort and get the threshold
    copy_of_X = X.copy()
    max_benefit_feature_col = copy_of_X[:, attribute_max_benefit]

    # sort the feature column first


    max_benefit_feature_col.sort()

    # print("max benefir column "+str(max_benefit_feature_col))

    # getting maximum benefit value from the feature column using the index that we calculated earlier
    # print("feature number, position "+str(attribute_max_benefit)+" "+str(max_benefit_indices_for_every_feature[col + 1]))

    # print("aa"+str(index_of_thres_with_max_benefit.shape))

    max_benefit_val = max_benefit_feature_col[int(index_of_thres_with_max_benefit+1)]

    # reset two arrays
    # maintain an array for Benefit values for all features
    benefit_array_for_one_feature.fill(0)
    # maintain an array for indices for the threshold values that give the maximum benefit for all features
    max_benefit_indices_for_every_feature.fill(0)

    #     print("\nMaximum Benefit Feature's position is: "+str(attribute_max_benefit+1))
    #     print("\nMaximum Benefit Column's values are: "+str(max_benefit_feature_col))
    #     print("\nMaximum Benefit Value: "+str(max_benefit_val))
    #     print("\nMaximum Benefit Feature's sorted position within the column:"+str(index_of_thres_with_max_benefit))
    # pass the column no of max benefit, value of the threshold with which to compare and the dataset itself

    return {'index': attribute_max_benefit + 1, 'value': max_benefit_val,
            'groups': test_split(attribute_max_benefit + 1, max_benefit_val, dataset)}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[0] for row in group]
    #print("\noutcome values for this terminal group: "+str(outcomes))
    p = max(set(outcomes), key=outcomes.count)
    #print("So max val of outcomes is "+(str(p)))
    return p
 
#recursive implementation of splitting
def split(node,max_depth,min_size,depth):
    left, right = node['groups']
    
    #print("left size "+ str(len(left)))
    #print("right size "+ str(len(right)))
    del (node['groups'])
    
    #check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
     
        
        
        
    # check for max depth
    if depth >= max_depth:
        #print("\nDepth reached max depth\n\n")
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
 
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
#         print("\nCame to left ")
#         print("left ko size is "+str(len(left)))
        node['left'] = get_split(left)
        #didnot reach here from the previous line, well it reaches now
        #print("what is received as left is : "+str(node['left']))
        split(node['left'], max_depth, min_size, depth + 1)  # process right child
    
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)
 
 
# Build a decision tree
def build_tree(csv_content, max_depth, min_size):
    print("\nMax Depth to use is "+(str(max_depth)))
    max_depth_list.append(max_depth)
    root = get_split(csv_content)
    split(root, max_depth, min_size, 1)
    return root

# predict with a decision tree
def predict(node, row):

    if row[node['index']] >= node['value']:

        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

    else:

        if isinstance(node['left'], dict):

            return predict(node['left'], row)
        else:
            return node['left']

 
 
#tree
def decision_tree(train, test, max_depth, min_size):
    
    #build a tree using training data
    
    for max_depth_val in np.linspace(1, max_depth+1,3,dtype=int):
        max_depth_to_use = max_depth_val


        tree = build_tree(train, max_depth_to_use, min_size)

        test_len = len(test)
        #print(tree)

        #create lists to store your predictions for the training as well as testing data
        predictions_train = list()
        predictions_test = list()

        #plt.figure()


        #predict using the created tree
        for row in train:
            prediction = predict(tree, row)
            predictions_train.append(prediction)

    #     print("pred y "+str(predictions_train))
    #     print("train y "+str(train[:,0]))
        training_data_accu = accuracy_metric(predictions_train,train[:,0])
        training_accuracy_list.append(training_data_accu)
        #print("\nTraining Data Accuracy is "+str(training_data_accu))
        if training_data_accu == 100:
            print("\nTraining Accuracy is 100% at depth "+str(max_depth_to_use))




        for row in test:
            prediction = predict(tree, row)
            predictions_test.append(prediction)
        testing_data_accuracy = accuracy_metric(predictions_test,test[:,0])
        testing_accuracy_list.append(testing_data_accuracy)





    
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if int(actual[i]) == int(predicted[i]):
            correct += 1
    return correct / float(len(actual)) * 100.0    


if __name__ == '__main__':

    Y, X = open_csv("pa3_train_reduced")
    train = np.column_stack((Y, X))
    Y, X = open_csv("pa3_valid_reduced")
    test = np.column_stack((Y, X))
    max_depth = 20
    min_size = 1

    decision_tree(train, test, max_depth, min_size)

    plt.plot(max_depth_list, training_accuracy_list, 'g--', label="Training Accuracy")
    plt.plot(max_depth_list, testing_accuracy_list, 'r--', label="Validation Accuracy")
    plt.legend()

    print("\nThe best validation accuracy is " + str(max(testing_accuracy_list)))
    plt.show()
    print("\nThe time taken for the whole execution is " + str(time.time() - start_time))







