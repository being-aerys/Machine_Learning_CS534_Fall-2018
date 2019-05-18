import csv
import numpy as np
#import matplotlib.pyplot as plt
import math



learner = 0
size_of_training_examples =4888
max_size_of_learner = 20


# def open_csv(filename):
#     f = open(filename + ".csv", 'r')
#     reader = csv.reader(f)
#     labels = []
#     features = []

#     for i, row in enumerate(reader):
#         labels.append(float(row[0]))
#         features.append([float(x) for x in row[1:]])
#     features = numpy.array(features)
#     labels = numpy.array(labels)

#     labels[labels == 3.0] = 1
#     labels[labels == 5.0] = -1

#     return labels, features


# Gini index all the data
def gini_data(y, weights):
    # count the Y equal to one
    c_root_pos = 0
    index_of_pos_elements = np.nonzero(y == 1.0)[0]
    c_root_pos = 0
    for x in index_of_pos_elements:
        c_root_pos += weights[x]

    c_root_total = np.sum(weights)

    #     print("c_root_positive "+str(c_root_pos))
    #     print("c root total "+str(c_root_total))

    return (1 - (c_root_pos / c_root_total) ** 2 - ((c_root_total - c_root_pos) / c_root_total) ** 2)


def get_split(dataset, weights):
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

    for col in range(X.shape[1]):

        single_feature = X[..., col]
        # print("size of single feature "+(str(single_feature.shape)+"\n\n"))

        # stack y_to_take and X[col]
        stacked_feature = np.column_stack((Y, single_feature))
        stacked_weights = np.column_stack((weights, single_feature))
        # pass as arg the indices of the sorted 2nd column
        sorted_feature = stacked_feature[np.argsort(stacked_feature[:, 1])]
        sorted_weights = stacked_weights[np.argsort(stacked_weights[:, 1])]
        y_to_take = sorted_feature[..., 0]
        weights_to_take = sorted_weights[..., 0]
        # print("y_to_take",y_to_take)
        # gives the positions of the x values where the values change
        indices_where_change = np.where(y_to_take[:-1] != y_to_take[1:])[0]
        # print(str(indices_where_change))

        # just a redundant thing
        indices_where_Actual_thresholds_are = indices_where_change  # print("y to take \n"+str(y_to_take))
        # print("indices change see \n\n"+str(indices_where_Actual_thresholds_are))

        # lets iterate for each position within a feature  where there is a change in y lable after we have sorted x values
        for i in range(len(indices_where_Actual_thresholds_are)):
            # y_*weights

            index_of_pos_elements = np.nonzero(y_to_take[:indices_where_Actual_thresholds_are[i] + 1] == 1.0)[0]
            # print("index of thresh=", indices_where_Actual_thresholds_are[i] + 1)
            # print("indices of positions of elements", index_of_pos_elements)

            CL_pos = 0
            for x in index_of_pos_elements:
                # print("x",x)
                # print("ind",x)
                CL_pos += weights_to_take[x]

            # print("non zero count is "+str(CL_pos))
            CL_total = np.sum(weights_to_take[:indices_where_Actual_thresholds_are[i] + 1])

            CL_neg = CL_total - CL_pos
            p_plus = CL_pos / (CL_total)
            p_neg = CL_neg / (CL_total)
            UAL = 1 - (p_plus) ** 2 - (p_neg) ** 2

            ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
            # negative part
            index_of_pos_elements = np.nonzero(y_to_take[indices_where_Actual_thresholds_are[i] + 1:] == 1.0)[0]
            # pos_element_right = [weights[i] for i in index_of_pos_elements]
            # print("index of thresh=", indices_where_Actual_thresholds_are[i] + 1)
            # print("indices of positions of elements", index_of_pos_elements)
            # print(weights_to_take)
            CR_pos = 0
            for x in index_of_pos_elements:
                CR_pos += weights_to_take[
                    indices_where_Actual_thresholds_are[i] + x + 1]  # print(indices_where_Actual_thresholds_are[i]+x+1)
            CR_total = np.sum(weights_to_take[indices_where_Actual_thresholds_are[i] + 1:])
            CR_neg = CR_total - CR_pos
            p_plus_r = CR_pos / (CR_total)
            p_neg_r = CR_neg / (CR_total)
            UAR = 1 - (p_plus_r) ** 2 - (p_neg_r) ** 2
            # print("UAR",str(UAR))
            # probabilities left righht
            pl = (CL_total) / np.sum(weights_to_take)
            pr = (CR_total) / np.sum(weights_to_take)

            # Benefit of splitting at this threshold i (length of this value varies among different features)
            Benefit_of_split_at_this_i = (gini_data(y_to_take, weights_to_take) - pl * UAL - pr * UAR)
            # Store the benefit of splitting at this threhold if it exceeds the previously stored value for this feature
            if (Benefit_of_split_at_this_i > benefit_array_for_one_feature[col + 1]):
                # print("benefit grater at i, threshold pos "+str(Benefit_of_split_at_this_i)+ " "+str(i))
                benefit_array_for_one_feature[col + 1] = Benefit_of_split_at_this_i
                # also store the indices for these max benefit positions-- for each col value, we have a different value
                max_benefit_indices_for_every_feature[col + 1] = indices_where_Actual_thresholds_are[i]

    attribute_max_benefit = np.argmax(benefit_array_for_one_feature[1:])

    index_of_thres_with_max_benefit = max_benefit_indices_for_every_feature[attribute_max_benefit + 1]

    copy_of_X = X.copy()
    max_benefit_feature_col = copy_of_X[:, attribute_max_benefit]

    # sort the feature column first

    max_benefit_feature_col.sort()

    max_benefit_val = max_benefit_feature_col[int(index_of_thres_with_max_benefit) + 1]
    # print("attribute max=",attribute_max_benefit)
    # reset two arrays
    # maintain an array for Benefit values for all features
    benefit_array_for_one_feature.fill(0)
    # maintain an array for indices for the threshold values that give the maximum benefit for all features
    max_benefit_indices_for_every_feature.fill(0)

    return {'index': attribute_max_benefit + 1, 'value': max_benefit_val,
            'groups': test_split(attribute_max_benefit + 1, max_benefit_val, dataset, weights)}


# split
def test_split(index, value, dataset, weights):
    left_dataset, right_dataset = list(), list()
    weights_left, weights_right = list(), list()
    # print("index of feature is "+str(index))
    for j, row in enumerate(dataset):
        if row[index] < value:
            # print(" left \n"+str(row[index]))
            left_dataset.append(row)
            weights_left.append(weights[j])
        else:
            # print(" right \n"+str(row[index]))
            right_dataset.append(row)
            weights_right.append(weights[j])

    return left_dataset, right_dataset, weights_left, weights_right


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[0] for index_row, row in enumerate(group)]
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
    left, right, weights_left, weights_right = node['groups']

    del (node['groups'])

    # check for a no split
    if not left or not right:
        #print("left or right empty !@@@@@@@@@@@@@@@@@@@@@@@@@@!")
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # check for max depth
    if depth >= max_depth:
        #print("max depth is reached!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        node['left'], node['right'] = to_terminal(left), to_terminal(right)

        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)

    else:
        node['left'] = get_split(left, weights_left)
        # didnot reach here from the previous line, well it reaches now
        # print("what is received as left is : "+str(node['left']))
        split(node['left'], max_depth, min_size, depth + 1)  # process right child

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, weights_right)
        split(node['right'], max_depth, min_size, depth + 1)  # print(depth)


# Build a decision tree
def build_tree(csv_content, max_depth, min_size, weights):
    root = get_split(csv_content, weights)
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


# tree
def decision_tree(train, test, max_depth, min_size, weights):
    #print("decision tree once")
    # build a tree using training data
    tree = build_tree(train, max_depth, min_size, weights)
    # create lists to store your predictions for the training as well as testing data
    prediction_train = []
    prediction_test = []

    # training
    for row in train:
        prediction = predict(tree, row)
        prediction_train.append(prediction)
    train_data_accuracy = accuracy_metric(prediction_train, train[:, 0])

    # validation
    for row in test:
        prediction = predict(tree, row)
        prediction_test.append(prediction)
    testing_data_accuracy = accuracy_metric(prediction_test, test[:, 0])

    #error = np.sum(prediction_test == test[:, 0]) / len(test)
    error = (100-testing_data_accuracy)/100
    #print("total error os " + str(error))
    return testing_data_accuracy, error, prediction_test,train_data_accuracy


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
        if (i < size_of_training_examples):
            labels.append(float(row[0]))
            features.append([float(x) for x in row[1:]])

    features = np.array(features)
    labels = np.array(labels)

    labels[labels == 3.0] = 1
    labels[labels == 5.0] = -1

    return labels, features


# vector_for_prodcuts_of_alphas_and_predictions = np.array([])
def final_hypothesis(alphas_for_a_learner):
    alpha_array = np.array(alphas_for_all_learners)
    predictions_array = np.array(matrix_to_store_predictions_for_All_learners)
    print("shape of alpha array and predictions array " + str(alpha_array.shape) + " " + str(predictions_array.shape))
    print("shape of alphas for all learners " + str(alphas_for_a_learner))
    print("shape of matrix_to_store_predictions_for_All_learners" + str(matrix_to_store_predictions_for_All_learners))
    #     print("vector "+str(vector_for_prodcuts_of_alphas_and_predictions.shape))
    vector_for_prodcuts_of_alphas_and_predictions = np.array([4888, 4])

    for i in range(0, 4888):
        vector_for_prodcuts_of_alphas_and_predictions[i, learner] = alphas_for_a_learner[i] * \
                                                                    matrix_to_store_predictions_for_All_learners[...,]
        print("here we " + str(vector_for_prodcuts_of_alphas_and_predictions.shape))
    # final_prediction = np.sum(vector_for_prodcuts_of_alphas_and_predictions)
    learner += 1


#     print("\nFinal Prediction is "+str(vector_for_prodcuts_of_alphas_and_predictions))


# upload the dataset
Y_train, X_train = open_csv("pa3_train_reduced")

Y_test, X_test = open_csv("pa3_valid_reduced")
train_dataset = np.column_stack((Y_train, X_train))
test_dataset = np.column_stack((Y_test, X_test))
max_depth = 9
min_size = 1

number_of_example_train = len(Y_train)
number_of_example_valid = len(Y_test)

# print("weights of examples is  "+str(weights_of_examples))
# node=build_tree(train_dataset,max_depth,min_size,weights)
#
# print(node)
# accuracy,error,prediction_test=decision_tree(train_dataset,test_dataset,max_depth,min_size,weights)


# print("validation set accuracy",accuracy)
# print("prediction",prediction_test)


values_returned_on_each_call_of_decision_tree = []
alpha_for_each_learner = np.zeros(4)
# matrix_to_store_predictions_for_All_learners = np.zeros((4888,10))
matrix_to_store_predictions_for_All_learners = np.zeros((4888, max_size_of_learner))

learners = [1, 5, 10, 20]


validation_accuracy=[]
training_accuracy=[]
print("L= [1 ,5,10,20]...:")
for j in learners:
    print("Adaboost with L="+str(j))
	
	# set weights for exampls for each learner
    weights_of_examples = np.ones(number_of_example_train, dtype=float) / number_of_example_train
    # set alpha for this learner
    alpha_for_each_learner = np.zeros(j)
    predictions_times_alpha_array = np.zeros(number_of_example_train,dtype=float)

    #print("j", str(j))
    for i in range(0, j):
        # learn a tree

        print("We are at iteration.... " + str(i))
        # gives us the training and testing accuracy for all the trees
        # print("old weights " + str(weights_of_examples))
        value_returned_as_dict = decision_tree(train_dataset, test_dataset, max_depth, min_size, weights_of_examples)

          # error predictions on valid data
        test_data_acc=value_returned_as_dict[0];print("test accuracy for classifier, "+str(i)+"\t"+str(test_data_acc))
        error_for_this_classifier = value_returned_as_dict[1]
        predicted_values_by_this_classifier = value_returned_as_dict[2]

        train_data_accuracy=value_returned_as_dict[3]
        print("training accu="+"\t"+str(train_data_accuracy))
        # print("training acc#############################",train_data_accuracy)
        # print("###############################################################")
        predicted_values_by_this_classifier=np.array(predicted_values_by_this_classifier,dtype=float)
        # print("pred "+str(predicted_values_by_this_classifier))

        #         values_returned_on_each_call_of_decision_tree.append(decision_tree(train, test, depth, min_size,weights_of_examples))

        prob=(1 - error_for_this_classifier) / error_for_this_classifier
        print("error_for_this_classifier=",error_for_this_classifier)
        number = round(prob, 3)
        print("log############",number)
        alpha_for_each_learner[i] = ((0.5) * math.log(number))
        # print("predicted val "+str(predicted_values_by_this_classifier))
        # print("predicted val "+str(size_of_training_examples))

        predictions_times_alpha_array = predictions_times_alpha_array + np.multiply(float(alpha_for_each_learner[i]) , predicted_values_by_this_classifier)
        #print("pred_times_value",predictions_times_alpha_array)

        #         print("here hai " + str(matrix_to_store_predictions_for_All_learners))

        #       mistaken predictions

        indices_where_predictions_are_different_than_truth = np.nonzero(predicted_values_by_this_classifier != test_dataset[:, 0])

        #         correct predictions
        indices_where_predictions_Are_the_same = np.nonzero(predicted_values_by_this_classifier == test_dataset[:, 0])
        #print("weights before",weights_of_examples[indices_where_predictions_are_different_than_truth])
        for jj in range(0, len(indices_where_predictions_are_different_than_truth)):
            #print("oooo"+ str(predicted_values_by_this_classifier)+"hhhhh"+str(test_dataset[:, 0]))
            # print("jeje"+str(indices_where_predictions_are_different_than_truth[i]))

            weights_of_examples[indices_where_predictions_are_different_than_truth[jj]] = weights_of_examples[
                                                                                             indices_where_predictions_are_different_than_truth[
                                                                                                 jj]] * math.exp(alpha_for_each_learner[i])

            # print("---- after ",indices_where_predictions_are_different_than_truth[jj])
        # print("hhhh"+str( weights_of_examples[indices_where_predictions_are_different_than_truth]))

        for k in range(0, len(indices_where_predictions_Are_the_same)):
            # print("jeje"+str(indices_where_predictions_Are_the_same[j]))
            a = -1 * alpha_for_each_learner[k]

            weights_of_examples[indices_where_predictions_Are_the_same[k]] = weights_of_examples[
                                                                                 indices_where_predictions_Are_the_same[
                                                                                     k]] * (math.exp(a))

        # print("new  weight of examples  "+str(weights_of_examples))
        weights_of_examples = weights_of_examples / np.sum(weights_of_examples)
        #print("new  weight of examples  " + str(weights_of_examples))

        # predict using this learner  # print(" all alpohas are  "+str(alpha_for_each_learner))

    # final_classification(vector_for_prodcuts_of_alphas_and_predictions)
    predicted_value_by_ada_learner=(np.sign((predictions_times_alpha_array)))
    #print("pewd vL BY LWrne "+str(predicted_value_by_ada_learner))
    validation_accuracy.append(accuracy_metric(predicted_value_by_ada_learner, test_dataset[:, 0]))
    training_accuracy.append(train_data_accuracy)
itr_list=[1,5,10,20]
#plt.plot(itr_list,validation_accuracy,'g--',label = "Training Accuracy")
# plt.scatter(itr_list, train_acc_list, color = 'blue', s = 15)
# blue_line, = plt.plot(itr_list, train_acc_list, color = 'blue', label = 'Training Accuracy')
#plt.title("Number of Weak Learners vs Accuracy")
#plt.xlabel("Number of Weak Learners Used")
#plt.ylabel("Accuracy Percentage")
#plt.show()
print("Valiadtion accrucy")
print(validation_accuracy)
print("training accuracy")
print(training_accuracy)