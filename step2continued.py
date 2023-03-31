import math
import numpy
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import stats
from numpy import nan
def process_data(valueis):
            data111 = '/Users/nilanjanadas/Downloads/psy_2/Pupil_Data/'+valueis
            data222 = '/Users/nilanjanadas/Downloads/psy_2/Behavioral_Data/'+valueis
            liststart, listend, data, diff, inco = [], [], [],[], []
            with open(data222) as file_obj:
                reader_obj = csv.reader(file_obj)
                next(reader_obj, None)
                for row in reader_obj:
                    start = float(row[0])
                    newVal = row[1]
                    if row[1] == 'NaN':
                        newVal = 8000
                    end = float(row[0]) + float(newVal)
                    liststart.append(start)
                    listend.append(end)
                    diff.append(float(row[2]))
                    inco.append(float(row[3]))
            stimpupil, data1 = [], []
            with open(data111) as file_obj:
                reader_obj1 = csv.reader(file_obj)
                next(reader_obj1, None)
                for row in reader_obj1:
                    for j in range(len(liststart)):
                        if float(row[0]) == int(float(liststart[j])):
                            stimpupil.append([j + 1, float(row[0]), float(row[1])])
                            data.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            data1.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            break
                        elif float(row[0]) == int(float(liststart[j])) + 1:
                            stimpupil.append([j + 1, float(row[0]), float(row[1])])
                            data.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            data1.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            break
                        elif float(row[0]) == int(float(liststart[j])) + 2:
                            stimpupil.append([j + 1, float(row[0]),float(row[1])])
                            data.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            data1.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            break
                        elif float(row[0]) == int(float(liststart[j])) + 3:
                            stimpupil.append([j + 1, float(row[0]), float(row[1])])
                            data.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            data1.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            break
                        elif float(row[0]) == int(float(liststart[j])) + 4:
                            stimpupil.append([j + 1,float(row[0]), float(row[1])])
                            data.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            data1.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            break
                        elif float(row[0]) >= float(liststart[j]) and float(row[0]) <= float(listend[j]):
                            data.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            data1.append([(j + 1), float(row[0]), row[1], diff[j], inco[j]])
                            break
            listcopy, stimpupil1, count, var = [], [], 0, 0
            stimpupil.sort()
            for l in range(len(stimpupil)):
                if l == 0:
                   stimpupil1.append(stimpupil[l])
                   var = stimpupil[l][0]
                   continue
                if stimpupil[l][0] != var:
                    stimpupil1.append(stimpupil[l])
                    var = stimpupil[l][0]

            for lst in data1:
              for j in stimpupil1:
                 if lst[0] == j[0] :
                   lst[1] = lst[1] - j[1]
                   lst[2] = float(lst[2]) - j[2]
                   break

            listplottime, listplotpupil, listplotpupil2 = [], [], []
            diff1, inco1, diff2, inco2, easytime, easypupil, hardtime, hardpupil, incorrecttime, incorrectpupil, correcttime, correctpupil = [], [], [], [], [], [], [], [], [],[],[],[]
            maxm = [float('-inf'), 0]
            for i in range(1, 41):
                listplottime1, listplotpupil1,diff1, inco1 =[], [], [],[]
                for j in data1:
                    if i == j[0] :
                        if j[1] > maxm[0]:
                            maxm[0] = j[1]
                            maxm[1] = i
                        listplottime1.append(j[1])
                        listplotpupil1.append(float(j[2]))
                        diff1.append(float(j[3]))
                        inco1.append(float(j[4]))
                listplottime.append(listplottime1)
                listplotpupil.append(listplotpupil1)
                copya = listplotpupil1.copy()
                listplotpupil2.append(copya)
                diff2.append(diff1)
                inco2.append(inco1)

            maxnew, num = float('-inf'), 0
            for i, avg in enumerate(listplotpupil2):
                if maxnew < len(avg):
                    maxnew = len(avg)
                    num = i
            for avg1 in listplotpupil2:
                if len(avg1) < maxnew:
                    sub = maxnew - len(avg1)
                    for i in range(sub):
                        avg1.append(0)
            et, ep, ht, hp, it, ip, ct, cp = [],[],[],[],[],[],[],[]

            for i, k in enumerate(diff2):
                if k[0] == 1.0:
                    copyd = listplotpupil2[i].copy()
                    ep.append(copyd)
                elif k[0] == 2.0:
                    copye = listplotpupil2[i].copy()
                    hp.append(copye)

            for i, k in enumerate(inco2):
                if k[0] == 0.0:
                    copyf = listplotpupil2[i].copy()
                    ip.append(copyf)
                elif k[0] == 1.0:
                    copyg = listplotpupil2[i].copy()
                    cp.append(copyg)

            return ep, hp, ip, cp, listplotpupil2, listplottime

def main():
    meganewList, meganewList1, meganewList2, meganewList3, meganewList4, megalistplottime = [], [], [], [], [], []
    for i in range(1, 53):
        print('Process started for subject ',i)
        word = 'subject_'+str(i)+'_behave.csv'
        ans = process_data(word)
        meganewList = meganewList + ans[0]
        meganewList1 = meganewList1 + ans[1]
        meganewList2 = meganewList2 + ans[2]
        meganewList3 = meganewList3 + ans[3]
        meganewList4 = meganewList4 + ans[4]
        megalistplottime = megalistplottime + ans[5]
    maxm = 0
    for i in meganewList:
        if maxm < len(i):
            maxm = len(i)
    for a in meganewList:

        if len(a) < maxm:
            sub = maxm - len(a)
            for i in range(sub):
                a.append(nan)
    for b in meganewList1:
                    if len(b) < maxm:
                        sub = maxm - len(b)
                        for i in range(sub):
                            b.append(nan)

    for c in meganewList2:
                   if len(c) < maxm:
                        sub = maxm - len(c)
                        for i in range(sub):
                            c.append(nan)

    for d in meganewList3:
                if len(d) < maxm:
                    sub = maxm - len(d)
                    for i in range(sub):
                        d.append(nan)

    for e in meganewList4:
                if len(e) < maxm:
                    sub = maxm - len(e)
                    for i in range(sub):
                        e.append(nan)

    maxm1, trial = 0, 0
    for j, i in enumerate(megalistplottime):
        if maxm1 < len(i):
            maxm1 = len(i)
            trial = j

    newList = numpy.asarray(meganewList4)
    newList = np.nanmean(newList, axis=0)

    newList1 = numpy.asarray(meganewList)
    newList1 = np.nanmean(newList1, axis=0)

    newList2 = numpy.asarray(meganewList1)
    newList2 = np.nanmean(newList2, axis=0)

    newList3 = numpy.asarray(meganewList2)
    newList3 = np.nanmean(newList3, axis=0)

    newList4 = numpy.asarray(meganewList3)
    newList4 = np.nanmean(newList4, axis=0)

    plt.figure()
    plt.plot(megalistplottime[trial], newList, color="blue", linewidth=1)
    plt.title('Average of all Pupil Responses for all Subjects')
    plt.xlabel('Time')
    plt.ylabel('Pupil Diameter')

    plt.figure()
    plt.plot(megalistplottime[trial], newList1, color="purple", linewidth=1)
    plt.plot(megalistplottime[trial], newList2, color="blue", linewidth=1)
    plt.title('Average of Pupil Responses for Easy vs Hard for all Subjects')
    plt.legend(['Easy', 'Hard'])
    plt.xlabel('Time')
    plt.ylabel('Pupil Diameter')

    plt.figure()
    plt.plot(megalistplottime[trial], newList3, color="green", linewidth=1)
    plt.plot(megalistplottime[trial], newList4, color="blue", linewidth=1)
    plt.title('Average of Pupil Responses for Incorrect vs Correct for all Subjects')
    plt.legend(['Incorrect', 'Correct'])
    plt.xlabel('Time')
    plt.ylabel('Pupil Diameter')
    plt.show()

if __name__ == "__main__":

    main()








