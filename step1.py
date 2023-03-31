import math
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import stats
import statistics

def standarddev(valueis):

        data111 = '/Users/nilanjanadas/Downloads/psy_2/Pupil_Data/' + valueis
        csv1 = pd.read_csv(data111)
        pupil_diameter = csv1['Pupil Diameter']
        t = csv1['Time']
        std = statistics.stdev(pupil_diameter)
        return std,pupil_diameter, t

def main():
    dev, subject, pupil, tm = [], [], [], []
    print('Please wait, the standard deviation of pupil diameters for each subject is being calculated and the plots will be displayed.......')
    for i in range(1, 53):
        word = 'subject_' + str(i) + '_behave.csv'
        ans = standarddev(word)
        dev.append(ans[0])
        [float(i) for i in ans[1]]
        [float(i) for i in ans[2]]
        pupil.append(ans[1])
        tm.append(ans[2])
    for i in range(1, 53):
        subject.append(i)

    csv1 = pd.read_csv('/Users/nilanjanadas/Downloads/psy_2/Pupil_Data/subject_3_behave.csv')
    pupil_diameter = csv1['Pupil Diameter']
    time = csv1['Time']
    liststart, listend = [], []

    with open('/Users/nilanjanadas/Downloads/psy_2/Behavioral_Data/subject_3_behave.csv') as file_obj:
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

    plt.figure()
    plt.vlines(x=subject, ymin=0.21, ymax=0.74, colors='orange')
    plt.plot(subject, dev, color="black", linewidth=2)
    plt.title('Standard Deviation of Pupil Diameters for each Subject')
    plt.xlabel('Subject')
    plt.ylabel('Standard Deviation')

    plt.figure()
    plt.vlines(x=liststart[:4], ymin=3.5, ymax=6.9, colors='Black')
    plt.vlines(x=listend[:4], ymin=3.5, ymax=6.9, colors='Green')
    plt.plot(time[:15000], pupil_diameter[:15000], color="blue", linewidth=1)
    plt.title('Start - Stop with respect to Pupil Diameters for Subject 3')
    plt.legend(['Stimulus Onset', 'Stop', 'Pupil Diameter'])
    plt.xlabel('Time')
    plt.ylabel('Pupil Diameter')
    plt.show()

    plt.figure()
    plt.plot(tm[0], pupil[0], color="blue", linewidth=1)
    plt.plot(tm[1], pupil[1], color="orange", linewidth=1)
    plt.plot(tm[2], pupil[2], color="green", linewidth=1)
    plt.plot(tm[3], pupil[3], color="purple", linewidth=1)

    plt.title('Pupil Diameter with respect to Time for Subjects 1-4')
    plt.legend(['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4'])
    plt.xlabel('Time')
    plt.ylabel('Pupil Diameter')
    plt.show()
if __name__ == "__main__":
    main()


