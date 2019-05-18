"""DT weak
    I put: D(t) or weights(t)

Output: Predictions vector + accuracy_on_training +accuracy _on_test

"""

import data
import numpy as np
import csv
import matplotlib.pyplot as plt


#Gini index all the data
def gini_data(y,weights):
    # count the Y equal to one
    index_of_pos_elements = np.nonzero(y==1.0)[0]
    # print(len(index_of_pos_elements))
    c_root_pos=0

    for x in index_of_pos_elements:
        c_root_pos += weights[x]
        # print("gini",x)

    c_root_total = np.sum(weights)

    # print("c_root_positive "+str(c_root_pos))
    #     print("c root total "+str(c_root_total))

    return (1 - (c_root_pos / c_root_total) ** 2 - ((c_root_total - c_root_pos) / c_root_total) ** 2)
#split the data
def get_split(dataset,weights):
    # maintain an array for Benefit values for all features
    benefit_array_for_one_feature = np.zeros(101)
    # maintain an array for indices for the threshold values that give the maximum benefit for all features
    max_benefit_indices_for_every_feature = np.zeros(101)
    nump = []
    for row in dataset:
        nump.append(row)
    # print(str(len(dataset)) + "values in the dataset in get split")
    dataset = np.array(nump)
    Y = dataset[..., 0]
    X = dataset[..., 1:]
    #print("weights before",weights)
    for col in range(X.shape[1]):

        single_feature = X[..., col]
        # print("size of single feature "+(str(single_feature.shape)+"\n\n"))

        # stack y_to_take and X[col]
        stacked_feature = np.column_stack((Y, single_feature))
        stacked_weights=np.column_stack((weights,single_feature))
        # pass as arg the indices of the sorted 2nd column
        sorted_feature = stacked_feature[np.argsort(stacked_feature[:, 1])]
        sorted_weights=stacked_weights[np.argsort(stacked_weights[:,1])]
        y_to_take = sorted_feature[..., 0]
        weights_to_take=sorted_weights[...,0]
        # print("y_to_take",y_to_take)
        # gives the positions of the x values where the values change
        indices_where_change = np.where(y_to_take[:-1] != y_to_take[1:])[0]
        # print(str(indices_where_change))

        # just a redundant thing
        indices_where_Actual_thresholds_are = indices_where_change  # print("y to take \n"+str(y_to_take))
        # print("indices change see \n\n"+str(indices_where_Actual_thresholds_are))

        # lets iterate for each position within a feature  where there is a change in y lable after we have sorted x values
        for i in range(len(indices_where_Actual_thresholds_are)):
            #y_*weights

            index_of_pos_elements=np.nonzero(y_to_take[:indices_where_Actual_thresholds_are[i] + 1]==1.0)[0]
            # print("index of thresh=", indices_where_Actual_thresholds_are[i] + 1)


            CL_pos=0
            for x in index_of_pos_elements:
                # print("x",x)
                # print("ind",x)
                CL_pos += weights_to_take[x]

            # print("non zero count is "+str(CL_pos))
            CL_total = np.sum(weights_to_take[:indices_where_Actual_thresholds_are[i] + 1])
            CL_neg = CL_total - CL_pos

            p_plus = float(CL_pos) / float(CL_total)
            p_neg = float(CL_neg) / float(CL_total)
            UAL = 1 - (p_plus) ** 2 - (p_neg) ** 2



            ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
            #negative part
            index_of_pos_elements = np.nonzero(y_to_take[indices_where_Actual_thresholds_are[i] + 1:] == 1.0)[0]
            #pos_element_right = [weights[i] for i in index_of_pos_elements]
            # print("index of thresh=", indices_where_Actual_thresholds_are[i] + 1)
            # print("indices of positions of elements", index_of_pos_elements)
            #print(weights_to_take)
            CR_pos=0
            for x in index_of_pos_elements:
                CR_pos +=weights_to_take[indices_where_Actual_thresholds_are[i]+x+1]
                # print(indices_where_Actual_thresholds_are[i]+x+1)
            CR_total = np.sum(weights_to_take[indices_where_Actual_thresholds_are[i] + 1:])
            CR_neg = CR_total-CR_pos
            p_plus_r = float(CR_pos) / float(CR_total)
            p_neg_r = float(CR_neg) / float(CR_total)
            UAR = 1 - (p_plus_r) ** 2 - (p_neg_r) ** 2
            # print("UAR",str(UAR))
            #probabilities left righht
            pl = (CL_total) / np.sum(weights_to_take)
            pr = (CR_total) / np.sum(weights_to_take)


            # Benefit of splitting at this threshold i (length of this value varies among different features)
            Benefit_of_split_at_this_i = (gini_data(y_to_take,weights_to_take) - pl * UAL - pr * UAR)
            # Store the benefit of splitting at this threhold if it exceeds the previously stored value for this feature
            if (Benefit_of_split_at_this_i > benefit_array_for_one_feature[col + 1]):
                # print("benefit grater at i, threshold pos "+str(Benefit_of_split_at_this_i)+ " "+str(i))
                benefit_array_for_one_feature[col + 1] = Benefit_of_split_at_this_i
                # also store the indices for these max benefit positions-- for each col value, we have a different value
                max_benefit_indices_for_every_feature[col + 1] = indices_where_Actual_thresholds_are[i]

    #print("weigt after:",weights)
    attribute_max_benefit = np.argmax(benefit_array_for_one_feature[1:])

    index_of_thres_with_max_benefit = max_benefit_indices_for_every_feature[attribute_max_benefit + 1]

    copy_of_X = X.copy()
    max_benefit_feature_col = copy_of_X[:, attribute_max_benefit]

    # sort the feature column first


    max_benefit_feature_col.sort()

    max_benefit_val = max_benefit_feature_col[int(index_of_thres_with_max_benefit)+1]
    # print("attribute max=",attribute_max_benefit)
    # reset two arrays
    # maintain an array for Benefit values for all features
    benefit_array_for_one_feature.fill(0)
    # maintain an array for indices for the threshold values that give the maximum benefit for all features
    max_benefit_indices_for_every_feature.fill(0)

    return {'index': attribute_max_benefit + 1, 'value': max_benefit_val,
            'groups': test_split(attribute_max_benefit + 1, max_benefit_val, dataset,weights)}


def test_split(index,value,dataset,weights):
    left_dataset,right_dataset=list(),list()
    weights_left,weights_right=list(),list()
    # print("index of feature is "+str(index))
    for j,row in enumerate(dataset):
        if row[index] < value:
            # print(" left \n"+str(row[index]))
            left_dataset.append(row)
            weights_left.append(weights[j])

        else:
            # print(" right \n"+str(row[index]))
            right_dataset.append(row)
            weights_right.append(weights[j])


    return left_dataset, right_dataset,weights_left,weights_right




def to_terminal(group):
    outcomes = [row[0] for index_row,row in enumerate(group)]
    #
    #
    # if sum(outcomes)>=0:
    #     predict=1
    # else:
    #     predict=-1
    predict = max(set(outcomes), key=outcomes.count)
    return predict

# recursive implementation of splitting
def split(node, max_depth, min_size, depth):
    left, right,weights_left,weights_right = node['groups']


    del (node['groups'])

    # check for a no split
    if not left or not right:
        #print("left or right empty !")
        node['left'] = node['right'] = to_terminal(left + right)
        return


    # check for max depth
    if depth >= max_depth:
        #print("max depth is reached!")
        node['left'], node['right'] = to_terminal(left), to_terminal(right)

        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)

    else:
        node['left'] = get_split(left,weights_left)
        # didnot reach here from the previous line, well it reaches now
        # print("what is received as left is : "+str(node['left']))
        split(node['left'], max_depth, min_size, depth + 1)  # process right child

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right,weights_right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(csv_content, max_depth,weights):
    root = get_split(csv_content,weights)
    split(root, max_depth, 1, 1)
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


# Build tree
def decision_tree(train,test, max_depth,weights):
    # build a tree using training data
    tree = build_tree(train, max_depth,weights)
    # create lists to store your predictions for the training as well as testing data
    prediction_train = []
    prediction_test=[]


    # training
    for row in train:
        prediction1 = predict(tree, row)
        prediction_train.append(prediction1)
    train_data_accuracy = accuracy_metric(prediction_train, train[:, 0])


    #validation
    for row in test:
            prediction2 = predict(tree, row)
            prediction_test.append(prediction2)
    testing_data_accuracy = accuracy_metric(prediction_test,test[:,0])

    #
    # error=np.sum(prediction_test!=test[:,0])/len(test)
    return train_data_accuracy,testing_data_accuracy,prediction_test

#Return prediction error on validation dataset
def error(prediction_test,Y_test):
    return np.sum(prediction_test!=Y_test[:,0])/len(Y_test)


#accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if int(actual[i]) == int(predicted[i]):
            correct += 1
    return correct / float(len(actual)) * 100.0


def open_csv(filename):
    f = open(filename + ".csv", 'r')
    reader = csv.reader(f)
    labels = []
    features = []

    for i, row in enumerate(reader):
        if (i<1000):
            labels.append(float(row[0]))
            features.append([float(x) for x in row[1:]])

    features = np.array(features)
    labels = np.array(labels)

    labels[labels == 3.0] = 1
    labels[labels == 5.0] = -1

    return labels, features



if __name__ == '__main__':
    ############################## upload the dataset#########################
    Y_train, X_train = open_csv("pa3_train_reduced")

    Y_test, X_test = open_csv("pa3_valid_reduced")
    train_dataset = np.column_stack((Y_train, X_train))
    test_dataset=np.column_stack((Y_test,X_test))
    max_depth = 9
    number_of_example_train = len(Y_train)
    number_of_example_valid = len(Y_test)

    ########################weights##########################################
    #weights = np.random.normal(0, .1, len(X_train))
    weights=np.ones(len(X_train))/len(X_train)
    # print(weights)
    ########################################################################
    #############################prediction test,###########################
    tr_accuracy_list=[]
    t_accuracy_list=[]
    depth_size=1
    for depth_size in np.linspace(1, max_depth+1,3,dtype=int):
        print("...............................depth size", depth_size + 1)
        tr_accuracy, t_accuracy,prediction_vector = decision_tree(train_dataset, test_dataset, depth_size, weights)
        tr_accuracy_list.append(tr_accuracy)
        t_accuracy_list.append(t_accuracy)
        print(np.sum(prediction_vector != Y_test) / len(Y_test))
    print("train set accuracy", tr_accuracy_list)
    print("test set accuracy", t_accuracy_list)  # print("prediction", prediction_test)
    # plt.plot(np.linspace(1, max_depth+1,3,dtype=int),tr_accuracy_list, 'g--', label="Training Accuracy")
    # plt.plot(np.linspace(1, max_depth+1,3,dtype=int),t_accuracy_list, 'r--', label="Validation Accuracy")
    # plt.legend()
    # plt.title(" Training vs validation accuracy")
    # plt.xlabel("depth of the tree ")
    # plt.show()

