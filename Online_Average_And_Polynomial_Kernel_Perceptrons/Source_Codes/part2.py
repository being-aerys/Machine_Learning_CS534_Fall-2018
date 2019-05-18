"""
Average perceptron
"""
import numpy as np
import sys
from processing import *
import matplotlib.pyplot as plt


class Perceptron:
    iters=None
    def __init__(self,iteration):
        self.iters=iteration

    def implementation(self):
        #train X,Y
        obj1=processing("pa2_train")
        X,Y=obj1.open_csv()
        w=np.zeros(X.shape[1])
        wb=np.zeros(X.shape[1])
        c=0
        s=0
        iter=0
        acc1=[]
        acc2=[]
        #valid X2,Y2
        obj2 = processing("pa2_valid")
        X2, Y2 = obj2.open_csv()
        while iter<iters:
            acc1_c=0
            for i,X_i in enumerate(X):
                ut=np.sign(np.dot(np.transpose(w),X_i))
                if Y[i]*ut<=0:
                    if s+c>0:
                        wb=(s*wb+c*w)/(s+c)
                    s=s+c
                    w=w+np.dot(Y[i],X_i)
                    c=0
                else:
                    #acc1_c=acc1_c+1
                    c=c+1

            for i,X_i in enumerate(X):
                ut=np.sign(np.dot(np.transpose(wb),X_i))
                if Y[i]*ut>0:
                    acc1_c = acc1_c + 1
            acc1.append(acc1_c)
            acc2_c=0

            for i,X2_i in enumerate(X2):
                ut=np.sign(np.dot(np.transpose(wb),X2_i))
                if Y2[i]*ut>0:
                    acc2_c=acc2_c+1
            acc2.append(acc2_c)
            iter=iter+1
        if c>0:
            wb=(s*wb+c*w)/(s+c)


        print("Accuracy:\n","Training acc:")
        print(np.divide(acc1,Y.shape[0]))
        print("###############################\n","valid acc:")
        print(np.divide(acc2,Y2.shape[0]))

        #PLOT

        plt.plot(np.divide(acc1,Y.shape[0]), label="iters=" + str(iters) + "-train")
        plt.plot(np.divide(acc2,Y2.shape[0]), label="iters=" + str(iters) + "-valid")
        plt.title(" train and validation accuracies versus the iteration number ")
        plt.xlabel("iterations")
        plt.ylabel("Acc")
        plt.legend()
        plt.show()






if __name__ == '__main__':
    try:
        #Get the numbber of iterations
        iters=int(sys.argv[1])

    except ValueError:
        print("Invalid input")
    except IndexError:
        print("missing argument in command")


    percep=Perceptron(iters)
    percep.implementation()




