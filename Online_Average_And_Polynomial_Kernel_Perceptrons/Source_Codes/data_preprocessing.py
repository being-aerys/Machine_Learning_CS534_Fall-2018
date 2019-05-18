#pre-processing
import csv
import numpy

class processing:


    def __init__(self,filename):
        self.file=filename

    def filename(self):
        print(self.file)

    def my_func(a):
        #func apply 1,-1
        for x in a:
            if x is 3.0:
                x = 1
            elif x is 5.0:
                x = -1
        return a

    def array_for(a):
        return numpy.array([processing.my_func(xi) for xi in a])

    def open_csv(self):
        f = open(self.file + ".csv", 'r')
        reader = csv.reader(f)
        output=[]
        features=[]

        for i,row in enumerate(reader):
            output.append(float(row[0]))
            features.append([float(x) for x in row[1:]])
        features=numpy.array(features)
        output=numpy.array(output)

        output[output==3]=1
        output[output==5]=-1
        features =numpy.insert(features, 0, 1.0, axis=1)

        return features,output

    def open_csv_for_test(self):
        f = open(self.file + ".csv", 'r')
        reader = csv.reader(f)
        features = []
        for i,row in enumerate(reader):
            features.append([float(x) for x in row[0:]])
        features=numpy.array(features)
        features = numpy.insert(features, 0, 1.0, axis=1)
        return features


# obj=processing("pa2_train")
# features,output=obj.open_csv()
# print(features,"\n")
# print(len(output))


