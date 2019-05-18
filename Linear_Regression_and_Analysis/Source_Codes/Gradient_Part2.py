import matplotlib
import tkinter
import sys
import matplotlib.pyplot as plt

from Data_processing import *

class Gradiant_lamda:
    lr=1e-6
    def __init__(self,lamdas,epsilon):
        self.lamdas=lamdas
        self.epsilon=epsilon

    def gradiant_L(self):

        obj = Data_processing("PA1_train")
        t, f = obj.open_csv()
        print(t)

        training_feature = obj.feature_normalization(f)
        X1 = training_feature[:, 0:training_feature.shape[1] - 1]
        print("The X1 is "+ str(X1))
        Y1 = training_feature[:, training_feature.shape[1] - 1]
        print("The Y1 is "+ str(Y1))
        obj=Data_processing("PA1_dev")
        t,f=obj.open_csv()
        dev_feature = obj.feature_normalization(f)
        X2 = dev_feature[:, 0:dev_feature.shape[1] - 1]
        Y2 = dev_feature[:, dev_feature.shape[1] - 1]

        iterr=0
        for key,lamda in enumerate(self.lamdas):
            SSE1_lamda = []
            SSE2_lamda = []
            w = np.zeros(training_feature.shape[1] - 1)
            temp1 = w
            temp1 = temp1[1:]
            Arr = np.array([0])
            Arr2 = np.concatenate((Arr,temp1), axis=0, out=None)
            w_lamda = Arr2
            while True:


                SSE1_lamda.append(np.sum(np.square(np.dot(X1,w)-Y1)))



                SSE2_lamda.append(np.sum(np.square(np.dot(X2,w)-Y2)))
                G = 2 * (np.dot((np.dot(w, np.transpose(X1))-Y1),X1)+  lamda*w_lamda )# derivative of J
                w = w - G * self.lr
                temp1 = w
                temp1 = temp1[1:]
                Arr2 = np.concatenate((Arr,temp1), axis=0, out=None)
                w_lamda = Arr2
                iterr = iterr + 1
                if (iterr % 1000) == 0:
                    print("iteration=",iterr)
                    print("Gradiant",np.linalg.norm(G))

                if np.linalg.norm(G) < epsilon or np.linalg.norm(G)>1e45 or iterr >= 10000:

                    if np.linalg.norm(G) < epsilon:
                        print("Converged because norm of gradient < epsilon")
                    if np.linalg.norm(G)>1e14:
                        print("Converged because norm of grad > 1e14")

                    print("The weights for which we stopped or converged are " + str(w))

                    #report the SSE function of number of iterations append list

                    SSE1_lamda.append(SSE1_lamda[iterr-1])
                    SSE2_lamda.append(SSE2_lamda[iterr-1])

                    #iterations[key]=iterr
                    print(SSE1_lamda)
                    plt.plot(SSE1_lamda,label=str(lamda) +" training error")
                    plt.plot(SSE2_lamda, label=str(lamda) +" validation error")
                    iterr=0
                    break


        plt.yscale('log')
        plt.legend()
        plt.title("training SSE function of iterations ")
        plt.show()

        # print("SSE training---------\n",SSE1_lamda,"\n")
        # print("SSE validation---------\n",SSE2_lamda,"\n")
        # print("Iterations ---------\n",iterations)



if __name__ == '__main__':
    lamdas = [float(i) for i in sys.argv[1].strip('[]').split(',')]
    # lamdas=[0]
    epsilon = float(sys.argv[2])
    Gradiant_lamda=Gradiant_lamda(lamdas,epsilon)
    Gradiant_lamda.gradiant_L()





