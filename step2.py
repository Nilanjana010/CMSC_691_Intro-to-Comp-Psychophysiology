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
            liststart, listend, data = [], [], []
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
                        break

            stimpupil, data1 = [], []
            with open(data111) as file_obj:
                reader_obj1 = csv.reader(file_obj)
                next(reader_obj1, None)
                for row in reader_obj1:
                        if float(row[0]) >= float(liststart[0]) and float(row[0]) <= float(listend[0]):
                            data.append([float(row[0]), row[1]])
                            data1.append([float(row[0]), row[1]])

            listm, listp = [], []
            for d in data1:
                listm.append(d[0])
                listp.append(float(d[1]))
            return liststart, listend, listp, listm

def main():
    meganewList, meganewList1, meganewList2, meganewList3, meganewList4 = [], [], [], [], []
    for i in range(1, 53):
        print('Process started for subject ',i)
        word = 'subject_'+str(i)+'_behave.csv'
        ans = process_data(word)
        meganewList = meganewList + ans[0]
        meganewList1 = meganewList1 + ans[1]
        meganewList2.append(ans[2])
        meganewList3.append(ans[3])

    plt.figure()
    plt.vlines(x=meganewList[:4], ymin=3.5, ymax=6.25, colors='purple')
    plt.vlines(x=meganewList1[:4], ymin=3.5, ymax=6.25, colors='chocolate')
    plt.plot(meganewList3[0], meganewList2[0], color="blue", linewidth=1)
    plt.plot(meganewList3[1], meganewList2[1], color="green", linewidth=1)

    plt.plot(meganewList3[2], meganewList2[2], color="darkmagenta", linewidth=1)
    plt.plot(meganewList3[3], meganewList2[3], color="deeppink", linewidth=1)
    #plt.plot(meganewList3[4], meganewList2[4], color="blue", linewidth=1)
    #plt.plot(meganewList3[5], meganewList2[5], color="blue", linewidth=1)
    plt.legend(['Stimulus Onset', 'Stop', 'Subject 1', 'Subject 2', 'Subject 3', 'Subject 4'])
    plt.title('Trial 1 for Subjects 1-4')
    plt.xlabel('Time')
    plt.ylabel('Pupil Diameter')
    plt.show()
if __name__ == "__main__":

    main()








