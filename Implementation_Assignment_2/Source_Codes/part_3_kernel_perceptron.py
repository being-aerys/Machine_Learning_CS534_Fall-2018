import numpy as np
from processing import *
import matplotlib.pyplot as plt


#def poly_kernel(x, y, p):
 #   return (1 + np.dot(np.transpose(X), X))**p

def gram(X, p):
    n_samples, n_features = X.shape
    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    # K = (1 + np.dot(np.transpose(X), X))** p
    for i in range(n_samples):
        for j in range(n_samples):
            #        #K[i,j] = self.kernel(X[i], X[j])
            K[i, j] = (1 + np.dot((X[i]), X[j]))**p
    return K

class KernelPerceptron(object):

    def __init__(self, T = 15):
        #self.kernel = kernel
        self.T = T




    def fit(self, X, y,xv,yv,Kt):
        n_samples, n_features = X.shape
        #v_samples, v_features = xv.shape
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        acc_train=[]
        acc_val=[]
        #print(K.shape)
        for t in range(self.T):
            acc_t=0
            for i in range(n_samples):
                u = np.sign(np.sum(Kt[:, i] * self.alpha * y))

               # if np.dot(y[i],u) <= 0:
                if u*y[i]<=0:
                    self.alpha[i] += 1.0
                    #acc_t+=1

            #acc_train.append((X.shape[0]-acc_t)/X.shape[0])

            #print(self.alpha)
            #print(self.alpha.shape)
            #print(i)
            #print("train_acc",acc_train,"iter", t,"p" , p)
            #plt.plot(t, acc_t, label="Train Acc ")


            # acc_v=0
            # for j in range(v_samples):
            #     av = self.alpha[-1629:]
            #     #print(av.shape)
            #     uv = np.sign(np.sum(Kv[:, j] * av * yv))
            #     if uv*yv[j]<=0:
            #         acc_v+=1

            #acc_val.append((xv.shape[0]-acc_v)/xv.shape[0])
            #print(j)
            #print("val_acc",acc_v,"iter", t,"p" , p)
            #plt.plot(t, acc_v, label="Val Acc ")

        # Support vectors
            sv = self.alpha > 1e-5
            ind = np.arange(len(self.alpha))[sv]
            self.nalpha = self.alpha[sv]
            self.sv = X[sv]
            self.sv_y = y[sv]

            acc_train.append((np.sum(kp.predict(X) == y))/len(y))
            acc_val.append((np.sum(kp.predict(xv) == yv))/len(yv))

            if p == 3 and t == 3:
                ## Best Prediction at p=3 and 4th iteration

                y_test = kp.predict(X_test)
                np.savetxt("kplabel.csv", y_test, delimiter=",")
                print("kplabel=", y_test)

        return acc_train, acc_val


    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.nalpha, self.sv_y, self.sv):
                s += a * sv_y * (1 + np.dot((X[i]), sv))**p
                y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape

        return np.sign(self.project(X))

#data = Preprocessing()
#X_train,y_train, X_test, y_test = data.preprocess()

obj1=processing("pa2_train")
X_train,y_train = obj1.open_csv()
obj2 = processing("pa2_valid")
X_val, y_val = obj2.open_csv()
obj3 = processing("pa2_test_no_label")
X_test = obj3.open_csv_for_test()

v_max = []

for p in [1,2,3,7,15]:

    Kt = gram(X_train, p)
    #Kv = gram(X_val, p)
    #kp = KernelPerceptron(poly_kernel(X_train,y_train,p), T=15)
    kp = KernelPerceptron()


    #acc_val = kp.predict(X_val)

    acc_train,acc_val = kp.fit(X_train, y_train,X_val, y_val,Kt)
    #correct = np.sum(y_predict == y_val)
    print ("##degree##\n",p,"\n")
    print("##train acc",acc_train,"\n")
    print("##val acc",acc_val,"\n")
    plt.figure()
    plt.plot(acc_train, 'b',label="Train Acc ")
    plt.plot(acc_val, 'r', label="Val Acc ")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Iterations for p =" + str(p))
    plt.legend()
    #plt.show()
    filename = 'iters v acc p' + str(p) + '.png'
    plt.savefig(filename)

    v_max.append(np.amax(acc_val))
    print("##Max Val Acc", v_max, "\n")




plt.figure()
plt.plot([1,2,3,7,15],v_max, 'b', label="Max Val Acc ")
plt.xlabel("Polynomial Degree")
plt.ylabel("Max Validation Accuracy")
plt.title("Validation Accuracy vs Polynomial Degree")
#plt.legend()
#plt.show()
plt.savefig('max val acc vs p.png')



