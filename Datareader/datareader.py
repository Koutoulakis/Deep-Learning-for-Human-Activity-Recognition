import numpy as np
import csv
import sys
import os
import h5py
import pandas as pd
import simplejson as json
import sqlite3
import copy

# structure followed in this file is based on : https://github.com/nhammerla/deepHAR/tree/master/data
# and https://github.com/IRC-SPHERE/sphere-challenge

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
            print('Done.')
        elif dataset == 'opp':
            f = h5py.File('opportunity.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            print('Done.')
        elif dataset == 'pa2':
            f = h5py.File('pamap2.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            print('Done.')
        elif dataset == "sph":
            f = h5py.File('sphere.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
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
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
               ]
        # print "cols",cols
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
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # print "cols",cols
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


    def readSphere(self):
        files = {
            'train': ['00001','00002', '00003', '00004', '00005', '00006', '00007', '00008'],
            'test' : [ '00009', '00010']
            # 'test': ['00011','00012','00013','00014','00015','00016','00017','00018','00019','00020','00021','00022','00023','00024','00025','00026','00027',
            # '00028','00029','00030','00031','00032','00033','00034','00035','00036','00037','00038','00039','00040','00041','00042','00043','00044','00045',
            # '00046','00047','00048','00049','00050','00051','00052','00053','00054','00055','00056','00057','00058','00059','00060','00061','00062','00063',
            # '00064','00065','00066','00067','00068','00069','00070','00071','00072','00073','00074','00075','00076','00077','00078','00079','00080','00081',
            # '00082','00083','00084','00085','00086','00087','00088','00089','00090','00091','00092','00093','00094','00095','00096','00097','00098','00099',
            # '00100','00101','00102','00103','00104','00105','00106','00107','00108','00109','00110','00111','00112','00113','00114','00115','00116','00117',
            # '00118','00119','00120','00121','00122','00123','00124','00125','00126','00127','00128','00129','00130','00131','00132','00133','00134','00135',
            # '00136','00137','00138','00139','00140','00141','00142','00143','00144','00145','00146','00147','00148','00149','00150','00151','00152','00153',
            # '00154','00155','00156','00157','00158','00159','00160','00161','00162','00163','00164','00165','00166','00167','00168','00169','00170','00171',
            # '00172','00173','00174','00175','00176','00177','00178','00179','00180','00181','00182','00183','00184','00185','00186','00187','00188','00189',
            # '00190','00191','00192','00193','00194','00195','00196','00197','00198','00199','00200','00201','00202','00203','00204','00205','00206','00207',
            # '00208','00209','00210','00211','00212','00213','00214','00215','00216','00217','00218','00219','00220','00221','00222','00223','00224','00225',
            # '00226','00227','00228','00229','00230','00231','00232','00233','00234','00235','00236','00237','00238','00239','00240','00241','00242','00243',
            # '00244','00245','00246','00247','00248','00249','00250','00251','00252','00253','00254','00255','00256','00257','00258','00259','00260','00261',
            # '00262','00263','00264','00265','00266','00267','00268','00269','00270','00271','00272','00273','00274','00275','00276','00277','00278','00279',
            # '00280','00281','00282','00283','00284','00285','00286','00287','00288','00289','00290','00291','00292','00293','00294','00295','00296','00297',
            # '00298','00299','00300','00301','00302','00303','00304','00305','00306','00307','00308','00309','00310','00311','00312','00313','00314','00315',
            # '00316','00317','00318','00319','00320','00321','00322','00323','00324','00325','00326','00327','00328','00329','00330','00331','00332','00333',
            # '00334','00335','00336','00337','00338','00339','00340','00341','00342','00343','00344','00345','00346','00347','00348','00349','00350','00351',
            # '00352','00353','00354','00355','00356','00357','00358','00359','00360','00361','00362','00363','00364','00365','00366','00367','00368','00369',
            # '00370','00371','00372','00373','00374','00375','00376','00377','00378','00379','00380','00381','00382','00383','00384','00385','00386','00387',
            # '00388','00389','00390','00391','00392','00393','00394','00395','00396','00397','00398','00399','00400','00401','00402','00403','00404','00405',
            # '00406','00407','00408','00409','00410','00411','00412','00413','00414','00415','00416','00417','00418','00419','00420','00421','00422','00423',
            # '00424','00425','00426','00427','00428','00429','00430','00431','00432','00433','00434','00435','00436','00437','00438','00439','00440','00441',
            # '00442','00443','00444','00445','00446','00447','00448','00449','00450','00451','00452','00453','00454','00455','00456','00457','00458','00459',
            # '00460','00461','00462','00463','00464','00465','00466','00467','00468','00469','00470','00471','00472','00473','00474','00475','00476','00477',
            # '00478','00479','00480','00481','00482','00483','00484','00485','00486','00487','00488','00489','00490','00491','00492','00493','00494','00495',
            # '00496','00497','00498','00499','00500','00501','00502','00503','00504','00505','00506','00507','00508','00509','00510','00511','00512','00513',
            # '00514','00515','00516','00517','00518','00519','00520','00521','00522','00523','00524','00525','00526','00527','00528','00529','00530','00531',
            # '00532','00533','00534','00535','00536','00537','00538','00539','00540','00541','00542','00543','00544','00545','00546','00547','00548','00549',
            # '00550','00551','00552','00553','00554','00555','00556','00557','00558','00559','00560','00561','00562','00563','00564','00565','00566','00567',
            # '00568','00569','00570','00571','00572','00573','00574','00575','00576','00577','00578','00579','00580','00581','00582','00583','00584','00585',
            # '00586','00587','00588','00589','00590','00591','00592','00593','00594','00595','00596','00597','00598','00599','00600','00601','00602','00603',
            # '00604','00605','00606','00607','00608','00609','00610','00611','00612','00613','00614','00615','00616','00617','00618','00619','00620','00621',
            # '00622','00623','00624','00625','00626','00627','00628','00629','00630','00631','00632','00633','00634','00635','00636','00637','00638','00639',
            # '00640','00641','00642','00643','00644','00645','00646','00647','00648','00649','00650','00651','00652','00653','00654','00655','00656','00657',
            # '00658','00659','00660','00661','00662','00663','00664','00665','00666','00667','00668','00669','00670','00671','00672','00673','00674','00675',
            # '00676','00677','00678','00679','00680','00681','00682','00683','00684','00685','00686','00687','00688','00689','00690','00691','00692','00693',
            # '00694','00695','00696','00697','00698','00699','00700','00701','00702','00703','00704','00705','00706','00707','00708','00709','00710','00711',
            # '00712','00713','00714','00715','00716','00717','00718','00719','00720','00721','00722','00723','00724','00725','00726','00727','00728','00729',
            # '00730','00731','00732','00733','00734','00735','00736','00737','00738','00739','00740','00741','00742','00743','00744','00745','00746','00747',
            # '00748','00749','00750','00751','00752','00753','00754','00755','00756','00757','00758','00759','00760','00761','00762','00763','00764','00765',
            # '00766','00767','00768','00769','00770','00771','00772','00773','00774','00775','00776','00777','00778','00779','00780','00781','00782','00783',
            # '00784','00785','00786','00787','00788','00789','00790','00791','00792','00793','00794','00795','00796','00797','00798','00799','00800','00801',
            # '00802','00803','00804','00805','00806','00807','00808','00809','00810','00811','00812','00813','00814','00815','00816','00817','00818','00819',
            # '00820','00821','00822','00823','00824','00825','00826','00827','00828','00829','00830','00831','00832','00833','00834','00835','00836','00837',
            # '00838','00839','00840','00841','00842','00843','00844','00845','00846','00847','00848','00849','00850','00851','00852','00853','00854','00855',
            # '00856','00857','00858','00859','00860','00861','00862','00863','00864','00865','00866','00867','00868','00869','00870','00871','00872','00873',
            # '00874','00875','00876','00877','00878','00879','00880','00881','00882']
        }

        label_map = [
            (0,  'a_ascend'),
            (1,  'a_descend'),
            (2,  'a_jump'),
            (3,  'a_loadwalk'),
            (4,  'a_walk'),
            (5,  'p_bent'),
            (6,  'p_kneel'),
            (7,  'p_lie'),
            (8,  'p_sit'),
            (9,  'p_squat'),
            (10, 'p_stand'),
            (11, 't_bend'),
            (12, 't_kneel_stand'),
            (13, 't_lie_sit'),
            (14, 't_sit_lie'),
            (15, 't_sit_stand'),
            (16, 't_stand_kneel'),
            (17, 't_stand_sit'),
            (18, 't_straighten'),
            (19, 't_turn')
        ]
        
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        # print ("label2id=",labelToId)
        idToLabel = [x[1] for x in label_map]
        # print ("idToLabel=",idToLabel)
        # colums of the merged file (ie video_hall+video_living+video_kitchen+accelerometer+maxarg_target)
        # the strict order is : accelerometer_data, video_hall_data, living_room_data, target_value
        cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]

        # cols_acceleration = [1, 2, 3, 4, 5, 6, 7]
        # cols_video = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        data = {dataset: self.readSphereFiles(files[dataset],cols,labelToId,idToLabel)
                for dataset in ('train', 'test')}

        return data, idToLabel

# partial code was taken from :https://github.com/IRC-SPHERE/sphere-challenge/blob/master/visualise_data.py
    def readSphereFiles(self, filelist, cols, labelToId,idToLabel):
        data = []
        labels = []

        mapping = {'a_ascend': 0, 'a_descend': 1, 'a_jump': 2, 'a_loadwalk': 3, 'a_walk': 4, 'p_bent': 5, 'p_kneel': 6, 'p_lie': 7, 'p_sit': 8,
             'p_squat': 9, 'p_stand': 10, 't_bend': 11, 't_kneel_stand': 12, 't_lie_sit': 13, 't_sit_lie': 14, 't_sit_stand': 15, 't_stand_kneel': 16, 
             't_stand_sit': 17,'t_straighten': 18, 't_turn': 19}

        # mapping2 = {0:'a_ascend',  1:'a_descend',  2:'a_jump',  3:'a_loadwalk',  4:'a_walk',  5:'p_bent',  6:'p_kneel',  7:'p_lie',  8:'p_sit',
        #       9:'p_squat',  10:'p_stand',  11:'t_bend',  12:'t_kneel_stand',  13:'t_lie_sit',  14:'t_sit_lie',  15:'t_sit_stand',  16:'t_stand_kneel', 
        #       17:'t_stand_sit', 18:'t_straighten', 19: 't_turn'}
        for i, filename in enumerate(filelist):
            path = './train/%s/'%filename
            meta_root = './metadata/'
            video_cols = json.load(open(os.path.join(meta_root, 'video_feature_names.json')))
            centre_2d = video_cols['centre_2d']
            bb_2d = video_cols['bb_2d']
            centre_3d = video_cols['centre_3d']
            bb_3d = video_cols['bb_3d']
            print('Reading file %d of %d'%(i+1,len(filelist)))
            meta = json.load(open(os.path.join(path, 'meta.json')))
            acceleration_keys = json.load(open(os.path.join(meta_root, 'accelerometer_axes.json')))
            rssi_keys = json.load(open(os.path.join(meta_root, 'access_point_names.json')))
            video_names = json.load(open(os.path.join(meta_root, 'video_locations.json')))
            pir_names = json.load(open(os.path.join(meta_root, 'pir_locations.json')))
            location_targets = json.load(open(os.path.join(meta_root, 'rooms.json')))
            activity_targets = json.load(open(os.path.join(meta_root, 'annotations.json')))
            
            accel = load_wearable(path,acceleration_keys,rssi_keys)
            vid = load_video(path,video_names)
            pir = load_environmental(path)
            annot = load_annotations(path)
            targ = load_targets(path)
            # accel = accel.dropna(how='any')
            # vid = pd.DataFrame(vid.items())
            # vid = pd.DataFrame.from_dict(orient='index',data = vid)
            # vid = vid.dropna(how='any')

            #we have read the whole train set for the current file
            #now we trim off all unlabeled target instances
            targ = targ.dropna(how='any')
            #we feel the accelerometer NaN values with zero (mean impute would not make much sense)
            accel = accel.fillna(0)
            # print(i)
            # print(filename)
            
            #we get the target label for each instance, which would be the argmax, of the targe probability distribution
            targLabel = copy.deepcopy(targ)
            targLabel.drop(targLabel.columns[[0, 1]], axis=1, inplace=True)
            #we create a target column, with the corresponding argmax targets
            targ['target'] = targLabel.idxmax(axis=1)
            #delete the probability distribution columns
            for activity in idToLabel:
            	del targ[activity]
            # print(targ)
            

            # print("accel")
            # print(accel.keys())
            # print("vid['hallway']")
            # print(vid['hallway'].keys())
            # print("theEND")

            accel.insert(0, 't', 0)
            accel['t'] = accel.index
            vid['hallway']['t']= vid['hallway'].index
            vid['living_room']['t'] = vid['living_room'].index 
            vid['kitchen']['t'] = vid['kitchen'].index 
            merged = pd.merge(accel,vid['hallway'],how='outer',on='t')
            merged = pd.merge(merged,vid['living_room'],how='outer',on='t')
            merged = pd.merge(merged,vid['kitchen'],how='outer',on='t')

            # print ("accel.shape")
            # print (accel.shape)
            # print ("vid[hallway].shape")
            # print (vid['hallway'].shape)
            # print ("vid[living_room].shape")
            # print (vid['living_room'].shape)
            # print ("vid[kitchen].shape")
            # print (vid['kitchen'].shape)

            # Rename the columns  appropriately
            merged.columns = ['time', 'x', 'y','z', 'Kitchen_AP', 'Lounge_AP', 'Upstairs_AP',
            'Study_AP',  'centre_2d_x_hall',  'centre_2d_y_hall' , 'bb_2d_br_x_hall'  ,'bb_2d_br_y_hall',
            'bb_2d_tl_x_hall' , 'bb_2d_tl_y_hall' , 'centre_3d_x_hall' , 'centre_3d_y_hall' , 'centre_3d_z_hall',
            'bb_3d_brb_x_hall' , 'bb_3d_brb_y_hall' , 'bb_3d_brb_z_hall' , 'bb_3d_flt_x_hall' , 'bb_3d_flt_y_hall',
            'bb_3d_flt_z_hall' , 'centre_2d_x_living' , 'centre_2d_y_living' , 'bb_2d_br_x_living' , 'bb_2d_br_y_living',
            'bb_2d_tl_x_living' , 'bb_2d_tl_y_living' , 'centre_3d_x_living' , 'centre_3d_y_living' , 'centre_3d_z_living',
            'bb_3d_brb_x_living' , 'bb_3d_brb_y_living' ,'bb_3d_brb_z_living' , 'bb_3d_flt_x_living' , 'bb_3d_flt_y_living',
            'bb_3d_flt_z_living' , 'centre_2d_x_kitchen' , 'centre_2d_y_kitchen' , 'bb_2d_br_x_kitchen' , 'bb_2d_br_y_kitchen',
            'bb_2d_tl_x_kitchen' , 'bb_2d_tl_y_kitchen' , 'centre_3d_x_kitchen' , 'centre_3d_y_kitchen' , 'centre_3d_z_kitchen' , 'bb_3d_brb_x_kitchen',
            'bb_3d_brb_y_kitchen' , 'bb_3d_brb_z_kitchen' , 'bb_3d_flt_x_kitchen' , 'bb_3d_flt_y_kitchen' , 'bb_3d_flt_z_kitchen',
            ]
            # pd.set_option('display.max_columns', 500)
            # print("merged.keys()")
            # print(merged.keys())
            # print("merged.shape")
            # print(merged.shape)
            # print("merged")
            # print(merged.ix[:5, :54])
            # print()

            #concatinate the target file labels and start,end tuples with the accelerometer timeseries.
            # print("going for the sql table creation")
            conn = sqlite3.connect(':memory:')
            targ.to_sql('targ',conn,index=True)
            merged.to_sql('merged',conn,index=True)
            # vid['hallway'].to_sql('hall',conn,index=True)
            # vid['living_room'].to_sql('living',conn,index=True)
            # vid['kitchen'].to_sql('kitchen',conn,index=True)
            # print("just did the sql table creation")

            
            qry = '''
            select 
            time, x, y,z, Kitchen_AP, Lounge_AP, Upstairs_AP,
            Study_AP,  centre_2d_x_hall,  centre_2d_y_hall , bb_2d_br_x_hall  ,bb_2d_br_y_hall,
            bb_2d_tl_x_hall , bb_2d_tl_y_hall , centre_3d_x_hall , centre_3d_y_hall , centre_3d_z_hall,
            bb_3d_brb_x_hall , bb_3d_brb_y_hall , bb_3d_brb_z_hall , bb_3d_flt_x_hall , bb_3d_flt_y_hall,
            bb_3d_flt_z_hall , centre_2d_x_living , centre_2d_y_living , bb_2d_br_x_living , bb_2d_br_y_living,
            bb_2d_tl_x_living , bb_2d_tl_y_living , centre_3d_x_living , centre_3d_y_living , centre_3d_z_living,
            bb_3d_brb_x_living , bb_3d_brb_y_living ,bb_3d_brb_z_living , bb_3d_flt_x_living , bb_3d_flt_y_living,
            bb_3d_flt_z_living , centre_2d_x_kitchen , centre_2d_y_kitchen , bb_2d_br_x_kitchen , bb_2d_br_y_kitchen,
            bb_2d_tl_x_kitchen , bb_2d_tl_y_kitchen , centre_3d_x_kitchen , centre_3d_y_kitchen , centre_3d_z_kitchen , bb_3d_brb_x_kitchen,
            bb_3d_brb_y_kitchen , bb_3d_brb_z_kitchen , bb_3d_flt_x_kitchen , bb_3d_flt_y_kitchen , bb_3d_flt_z_kitchen, targ.target

            from merged join targ on merged.time between targ.start and targ.end
            '''
            # pd.set_option('display.max_columns', 500)
            # print("doing the query")
            res = pd.read_sql_query(qry,conn)
            # print("query done")
            # print("res.shape")
            # print(res.shape)
            # print("res")
            # print(res.ix[:5, :100])
            
            res["target"].replace(mapping, inplace=True)
            res = res.fillna(0)
            # print("res_after_mapping")
            # print(res.ix[:5, :100])
            
            conn.close()
            
            
            for index, line in res.iterrows():
            	elem = []
            	for ind in cols:
            		elem.append(line[ind])
            	if sum([x=='NaN' for x in elem]) == 0:
            		data.append([float(x) / 1000 for x in elem[:-1]])
            		labels.append(labelToId[str(int(elem[-1]))])
            	
        
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}
    
# partial code was taken from :https://github.com/IRC-SPHERE/sphere-challenge/blob/master/visualise_data.py
def load_wearable(path,acceleration_keys,rssi_keys):
    accel_rssi = pd.read_csv(os.path.join(path, 'acceleration.csv'), index_col='t')
    acceleration = accel_rssi[acceleration_keys]
    rssi = pd.DataFrame(index=acceleration.index)
    for kk in rssi_keys:
        if kk in accel_rssi:
            rssi[kk] = accel_rssi[kk]
        
        else:
            rssi[kk] = np.nan
            accel_rssi[kk] = np.nan
    
    accel_rssi = accel_rssi
    return accel_rssi

def load_environmental(path):
    pir = pd.read_csv(os.path.join(path, 'pir.csv'))
    return pir

def load_video(path,video_names):
    video = dict()
    for location in video_names:
        filename = os.path.join(path, 'video_{}.csv'.format(location))
        video[location] = pd.read_csv(filename, index_col='t')
    return video

def load_annotations(path):
    num_annotators = 0
    
    annotations = []
    locations = []

    targets = None 

    targets_file_name = os.path.join(path, 'targets.csv')
    if os.path.exists(targets_file_name): 
        targets = pd.read_csv(targets_file_name)
    
    while True:
        annotation_filename = "{}/annotations_{}.csv".format(path, num_annotators)
        location_filename = "{}/location_{}.csv".format(path, num_annotators)
        
        if not os.path.exists(annotation_filename):
            break
        
        annotations.append(pd.read_csv(annotation_filename))
        locations.append(pd.read_csv(location_filename))
        
        num_annotators += 1
    
    annotations_loaded = num_annotators != 0
    return annotations
def load_targets(path):
    targets = None
    targets_file_name = os.path.join(path, 'targets.csv')
    if os.path.exists(targets_file_name): 
        targets = pd.read_csv(targets_file_name)
    return targets
if __name__ == "__main__":
    print('Reading %s ' % (sys.argv[1]))
    dr = data_reader(sys.argv[1])
