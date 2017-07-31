import numpy as np
import csv
import sys
import os
import h5py
import simplejson as json

# structure followed in this file is based on : https://github.com/nhammerla/deepHAR/tree/master/data

class data_reader:
    def __init__(self, dataset):
        if dataset == 'dap':
            self.data, self.idToLabel = self.readDaphnet()
            self.save_data(dataset)
        elif dataset =='opp':
            self.data, self.idToLabel = self.readOpportunity()
            self.save_data(dataset)
        elif dataset == 'pa2':
            self.data, self.idToLabel = self.readPamap2()
            self.save_data(dataset)
        elif dataset == 'sph':
            self.data, self.idToLabel = self.readSphere()
            self.save_data(dataset)
        else:
            print('Not supported yet')
            sys.exit(0)

    def save_data(self,dataset):
        if dataset == 'dap':
            f = h5py.File('daphnet.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            with open('daphnet.h5.classes.json', 'w') as f:
                f.write(json.dumps(self.idToLabel))
            print('Done.')
        elif dataset == 'opp':
            f = h5py.File('opportunity.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            with open('opportunity.h5.classes.json', 'w') as f:
                f.write(json.dumps(self.idToLabel))
            print('Done.')
        elif dataset == 'pa2':
            f = h5py.File('pamap2.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            with open('pamap2.h5.classes.json', 'w') as f:
                f.write(json.dumps(self.idToLabel))
            print('Done.')
        elif dataset == "sph":
            f = h5py.File('sphere.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            with open('sphere.h5.classes.json', 'w') as f:
                f.write(json.dumps(self.idToLabel))
            print('Done.')
        else:
            print('Not supported yet')
            sys.exit(0)

    @property
    def train(self):
        return self.data['train']

    @property
    def test(self):
        return self.data['test']

    def readPamap2(self):
        files = {
            'train': ['subject101.dat', 'subject102.dat','subject103.dat','subject104.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat'],
            'test': ['subject106.dat']
        }
        label_map = [
            # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            (9, 'watching TV'),
            (10, 'computer work'),
            (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            (18, 'folding laundry'),
            (19, 'house cleaning'),
            (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        print "id2label=",idToLabel
        cols = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
               ]
        print "cols",cols
        data = {dataset: self.readPamap2Files(files[dataset], cols, labelToId)
                for dataset in ('train', 'test')}
        return data, idToLabel

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            with open('./Protocol/%s' % filename, 'r') as f:
                #print "f",f
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    #print "line=",line
                    elem = []
                    #not including the non related activity
                    if line[1] == "0":
                        continue
                    # if line[10] == "0":
                    #     continue
                    for ind in cols:
                        #print "ind=",ind
                        # if ind == 10:
                        #     # print "line[ind]",line[ind]
                        #     if line[ind] == "0":
                        #         continue
                        elem.append(line[ind])
                    # print "elem =",elem
                    # print "elem[:-1] =",elem[:-1]
                    # print "elem[0] =",elem[0]
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[0]])
                        # print "[x for x in elem[:-1]]=",[x for x in elem[:-1]]
                        # print "[float(x) / 1000 for x in elem[:-1]]=",[float(x) / 1000 for x in elem[:-1]]
                        # print "labelToId[elem[0]]=",labelToId[elem[0]]
                        # print "labelToId[elem[-1]]",labelToId[elem[-1]]
                        # sys.exit(0)
        
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}

    def readDaphnet(self):
        files = {
            'train': ['S01R01.txt', 'S01R02.txt','S03R01.txt','S03R02.txt', 'S03R03.txt', 'S04R01.txt', 'S05R01.txt', 'S05R02.txt','S06R01.txt', 'S06R02.txt', 'S07R01.txt', 'S07R02.txt', 'S08R01.txt','S10R01.txt'],
            'test': ['S02R01.txt', 'S02R02.txt']
        }
        label_map = [
            (1, 'No freeze'),
            (2, 'freeze')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        print "id2label=",idToLabel
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print "cols",cols
        data = {dataset: self.readDaphFiles(files[dataset], cols, labelToId)
                for dataset in ('train', 'test')}
        return data, idToLabel

    def readDaphFiles(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            with open('./dataset/%s' % filename, 'r') as f:
                #print "f",f
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    #print "line=",line
                    elem = []
                    #not including the non related activity
                    if line[10] == "0":
                        continue
                    for ind in cols:
                        #print "ind=",ind
                        if ind == 10:
                            # print "line[ind]",line[ind]
                            if line[ind] == "0":
                                continue
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])
        
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}

    def readOpportunity(self):
        files = {
            'train': ['S1-ADL1.dat','S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL5.dat', 'S2-Drill.dat', 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL5.dat', 'S3-Drill.dat', 'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'],
            'test': ['S2-ADL3.dat', 'S2-ADL4.dat','S3-ADL3.dat', 'S3-ADL4.dat']
        }
        #names are from label_legend.txt of Opportunity dataset
        #except 0-ie Other, which is an additional label
        label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        cols = [
            37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58,63, 64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 80, 81, 82, 83, 84,
            89, 90, 91, 92, 93, 94, 95, 96, 97, 102, 103, 104, 105, 106, 107, 108,109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 249
            ]

        data = {dataset: self.readOpportunityFiles(files[dataset], cols, labelToId)
                for dataset in ('train', 'test')}

        return data, idToLabel

#this is from https://github.com/nhammerla/deepHAR/tree/master/data and it is an opportunity Challenge reader. It is a python translation one
#for the official one provided by the dataset publishers in Matlab.
    def readOpportunityFiles(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            with open('./dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}

if __name__ == "__main__":
    print('Reading %s ' % (sys.argv[1]))
    dr = data_reader(sys.argv[1])
