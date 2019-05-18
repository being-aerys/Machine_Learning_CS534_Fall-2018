#! python3
import numpy as np
import sys
from Data_processing import *
import matplotlib.pyplot as plt

class Gradiant:
    def __init__(self,learning_set,epsilon):
        self.learning_set=learning_set
        self.epsilon=epsilon

    def gradiant_E(self):
        SSE1_lr=[]
        SSE2_lr=[]
        iterations = []
        weights=[]
        obj=Data_processing("PA1_train")
        t, f = obj.open_csv()
        #f = np.delete(f, [3, 4, 7, 10, 15, 16], 1)
        training_feature=obj.feature_normalization(f)
        X1=training_feature[:,0:training_feature.shape[1]-1]
        Y1=training_feature[:,training_feature.shape[1]-1]


        obj=Data_processing("PA1_dev")
        t,f=obj.open_csv()
        #f = np.delete(f, [3, 4, 7, 10, 15, 16], 1)
        dev_feature = obj.feature_normalization(f)
        X2 = dev_feature[:, 0:dev_feature.shape[1] - 1]
        Y2 = dev_feature[:, dev_feature.shape[1] - 1]



        for key, lr in enumerate(self.learning_set):
            SSE1=[]
            SSE2=[]
            gradiand_norm = []
            w=np.zeros(training_feature.shape[1]-1)
            print("#####################Lambda=",lr)
            iterr = 0
            while True:
                #calculate the SSE for each training and vlidtion set
                #SSE1.append(np.sum(np.square(np.dot(X1,w)-Y1)))
                SSE2.append(np.sum(np.square(np.dot(X2,w)-Y2)))
                G = np.dot(np.dot(w, np.transpose(X1)) - Y1, X1)


                w = w - G * lr

                iterr = iterr + 1
                if(iterr%1000==0):
                    #print("weights", w)
                    # iteration number
                    #
                    print("iteration=", iterr)
                    print("Gradiant=", np.linalg.norm(G))



                if np.linalg.norm(G) < self.epsilon   or np.linalg.norm(G)>1e45 or iterr>=1000 :
                    #report the SSE function of number pf iterations append list
                    # SSE1_lr.append(SSE1)
                    # SSE2_lr.append(SSE2)
                    weights.append(w)
                    # iterations.append(iterr)
                    print("weights=",w)

                    plt.plot(SSE1,label="alpha="+str(lr)+"-training")
                    plt.plot(SSE2,label="alpha="+str(lr)+"-validation")
                    break

        plt.yscale("log")
        plt.legend()
        plt.title(" Sum square error on the training: ")
        plt.ylabel("SSE")
        plt.xlabel("iterations")
        plt.show()

        #print("SSE training---------\n",SSE1_lr,"\n")
        #print("SSE validation-------\n",SSE2_lr,"\n")









if __name__ == '__main__':
    try:
        #Get the learning set as List from input line command
        learning_set = [float(i) for i in sys.argv[1].strip('[]').split(',')]
        #Get epsilon as float from input line command
        epsilon = float(sys.argv[2])
    except ValueError:
        print("Invalid input")
    except IndexError:
        print("missing argument in command")






try:
    gradiant = Gradiant(learning_set, epsilon)
    gradiant.gradiant_E()

except NameError:
    print("Input does not follow instructions, please follow Readme file")
