import csv
import numpy as np


class Data_processing:

    file=None

    def __init__(self,filename):
        # Instance Variable
        self.file = filename

    def filename(self):
        print(type(self.file))

    def extract_date(self):
        with open(self.file+".csv", 'r') as infile:
            # read the file as a dictionary for each row ({header : value})
            reader = csv.DictReader(infile)
            data = {}
            for row in reader:
                for header, value in row.items():
                    try:
                        data[header].append(value)
                    except KeyError:
                        data[header] = [value]

        # extract the Date and split the date into M D Y
        result=[]
        date = data['date']
        s=[i.split('/') for i in date]
        for i,x in enumerate(s):
            element=list(map(int,[j for j in x]))
            result.append(element)
        result=np.array(result)
        M,D,Y = result.T

        return M,D,Y

    def open_csv(self):
        f = open(self.file + ".csv", 'r')
        reader = csv.reader(f)
        f_h = None
        features = []

        for (i, row) in enumerate(reader):
            if i is 0:
                f_h = row[3:]
            else:

                features.append([float(x) for x in row[3:]])


        features = np.array(features)
        month, day, year = self.extract_date()
        #Add dummy column
        f_h.insert(0,"dummy")
        features=np.insert(features, 0, 1.0, axis=1)
        #Add month
        month, day, year = self.extract_date()
        f_h.insert(1, "month")
        features = np.insert(features, 1, month, axis=1)
        #Add Day
        f_h.insert(2, "day")
        features = np.insert(features, 2, day, axis=1)
        #Add Year
        f_h.insert(3, "year")
        features = np.insert(features, 3, year, axis=1)
        f.close()
        return f_h, features

    def seperate_features(features):
        numeric_features = []
        catagorical_features = []
        for (j, i) in enumerate(features):
            catagorical_features.append((list(i[5:6]) + list(i[7:9])))
            numeric_features.append(list(i[:5]) + [i[6]] + list(i[9:]))

        catagorical_features = np.array(catagorical_features)
        numeric_features = np.array(numeric_features)
        return numeric_features, catagorical_features

    def feature_normalization(self,features):

        max_row = np.max(features, axis=0)
        min_row = np.min(features, axis=0)
        for i in range(1, features.shape[1]-1):
            features[:, i] = np.divide(features[:, i] - min_row[i], max_row[i] - min_row[i])

        return features

    def normalization(self):
        t,f=open(str(self.file))
        features = self.feature_normalization(f)
        return features


# obj=Data_processing("PA1_train")
#
# t,f_h=obj.open_csv()
# f_h=obj.feature_normalization(f_h)
#
# print(t)





