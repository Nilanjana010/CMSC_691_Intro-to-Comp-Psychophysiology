import math
import numpy
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import stats
from numpy import nan
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

def process_data(valueis):
            data111 = '/Users/nilanjanadas/Downloads/psy_2/Pupil_Data/'+valueis
            data222 = '/Users/nilanjanadas/Downloads/psy_2/Behavioral_Data/'+valueis
            liststart, listend, data, diff, inco, resp = [], [], [],[], [], []
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
                    resp.append(float(row[1]))
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
            lstx = []
            for i in range(0, 40):
                lstpupil, lsttm, maxm = [],[], float("-inf")
                for d in data1:
                    if d[0] == i + 1:
                        lstpupil.append(float(d[2]))
                        lsttm.append(d[1])
                p, q = scipy.signal.find_peaks(lstpupil)
                for j in p:
                   if lstpupil[j] > maxm:
                       maxm = lstpupil[j]
                       tm = lsttm[j]
                if maxm == float("-inf"):
                    maxm = 0.0
                latency = abs(stimpupil[i][1] - tm)
                lstx.append([latency,maxm])
            x = numpy.asarray(lstx)
            y1 = numpy.asarray(resp)
            y2 = numpy.asarray(diff)
            y3 = numpy.asarray(inco)

            return x, y1, y2, y3

def main():
    meganewListx = np.zeros((40, 2))
    meganewListy1 = np.zeros(40)
    meganewListy2 = np.zeros(40)
    meganewListy3 = np.zeros(40)
    print('\nPlease wait the process will be started for each of the subjects one by one..........\n')
    for i in range(1, 53):
        if i == 3:
            continue
        print('Process started for subject ',i)
        word = 'subject_'+str(i)+'_behave.csv'
        ans = process_data(word)
        meganewListx = np.concatenate((meganewListx, ans[0]), axis=0)
        meganewListy1 = np.concatenate((meganewListy1, ans[1]), axis=0)
        meganewListy2 = np.concatenate((meganewListy2, ans[2]), axis=0)
        meganewListy3 = np.concatenate((meganewListy3, ans[3]), axis=0)
    meganewListy1[np.isnan(meganewListy1)] = 0
    meganewListy2[np.isnan(meganewListy2)] = 0
    meganewListy3[np.isnan(meganewListy3)] = 0

    data111 = '/Users/nilanjanadas/Downloads/psy_2/TestData/testData_pupil_only.csv'
    data112 = '/Users/nilanjanadas/Downloads/psy_2/TestData/testData_pupil.csv'
    data222 = '/Users/nilanjanadas/Downloads/psy_2/TestData/testData_behave.csv'
    liststart, listend, data, resp = [], [], [], []
    with open(data222) as file_obj:
        reader_obj = csv.reader(file_obj)
        next(reader_obj, None)
        for row in reader_obj:
            start = float(row[0])
            newVal = 8000
            end = float(row[0]) + float(newVal)
            liststart.append(start)
            listend.append(end)
    stimpupil, data1 = [], []
    with open(data112) as file_obj:
        reader_obj1 = csv.reader(file_obj)
        next(reader_obj1, None)
        for row in reader_obj1:
            for j in range(len(liststart)):
                if float(row[0]) == int(float(liststart[j])):
                    stimpupil.append([j + 1, float(row[0]), float(row[1])])
                    data.append([(j + 1), float(row[0]), row[1]])
                    data1.append([(j + 1), float(row[0]), row[1]])
                    break
                elif float(row[0]) == int(float(liststart[j])) + 1:
                    stimpupil.append([j + 1, float(row[0]), float(row[1])])
                    data.append([(j + 1), float(row[0]), row[1]])
                    data1.append([(j + 1), float(row[0]), row[1]])
                    break
                elif float(row[0]) == int(float(liststart[j])) + 2:
                    stimpupil.append([j + 1, float(row[0]), float(row[1])])
                    data.append([(j + 1), float(row[0]), row[1]])
                    data1.append([(j + 1), float(row[0]), row[1]])
                    break
                elif float(row[0]) == int(float(liststart[j])) + 3:
                    stimpupil.append([j + 1, float(row[0]), float(row[1])])
                    data.append([(j + 1), float(row[0]), row[1]])
                    data1.append([(j + 1), float(row[0]), row[1]])
                    break
                elif float(row[0]) == int(float(liststart[j])) + 4:
                    stimpupil.append([j + 1, float(row[0]), float(row[1])])
                    data.append([(j + 1), float(row[0]), row[1]])
                    data1.append([(j + 1), float(row[0]), row[1]])
                    break
                elif float(row[0]) >= float(liststart[j]) and float(row[0]) <= float(listend[j]):
                    data.append([(j + 1), float(row[0]), row[1]])
                    data1.append([(j + 1), float(row[0]), row[1]])
                    break
    lstx = []
    for i in range(0, 40):
        lstpupil, lsttm, maxm = [], [], float("-inf")
        for d in data1:
            if d[0] == i + 1:
                lstpupil.append(float(d[2]))
                lsttm.append(d[1])
        p, q = scipy.signal.find_peaks(lstpupil)
        for j in p:
            if lstpupil[j] > maxm:
                maxm = lstpupil[j]
                tm = lsttm[j]
        if maxm == float("-inf"):
            maxm = stimpupil[i][2]
        latency = abs(stimpupil[i][1] - tm)
        lstx.append([latency, maxm])
    trialno, subjectno = [], []
    for k in range (1, 41):
        trialno.append(k)
    for k in range (1, 53):
        subjectno.append(k)

    stdlatency, stdpupil, latencymean, pupilmean = [], [], [], []
    size = meganewListx[:, :].shape

    for l in range(0, size[0], 40):
        a = np.std(meganewListx[i:i + 40, :1])
        b = np.std(meganewListx[i:i + 40, 1:2])
        c = np.mean(meganewListx[i:i + 40, :1])
        d = np.mean(meganewListx[i:i + 40, 1:2])
        stdlatency.append(a)
        stdpupil.append(b)
        latencymean.append(c)
        pupilmean.append(d)

    plt.figure()
    plt.plot(trialno, meganewListx[40:80, :1], color="blue", linewidth=1)
    plt.title('Latency for each trial for subject 1')
    plt.xlabel('Trial no.')
    plt.ylabel('Feature: Latency')

    plt.figure()
    plt.plot(trialno, meganewListx[40:80, 1:2], color="blue", linewidth=1)
    plt.title('Peak pupil response for each trial for subject 1')
    plt.xlabel('trial no.')
    plt.ylabel('Feature: Peak pupil response')

    plt.figure()
    plt.plot(subjectno, stdlatency, color="blue", linewidth=1)
    plt.title('Standard deviation of latency for all subjects')
    plt.xlabel('Subject no')
    plt.ylabel('Standard Deviation: Latency')

    plt.figure()
    plt.plot(subjectno, stdpupil, color="violet", linewidth=1)
    plt.title('Standard deviation of peak pupil response for all subjects')
    plt.xlabel('Subject no')
    plt.ylabel('Standard Deviation: Peak pupil response')

    plt.figure()
    plt.plot(subjectno, pupilmean, color="violet", linewidth=1)
    plt.title('Mean of peak pupil response for all subjects')
    plt.xlabel('Subject no.')
    plt.ylabel('Mean')

    plt.figure()
    plt.plot(subjectno, latencymean, color="blue", linewidth=1)
    plt.title('Mean of latency for all subjects')
    plt.xlabel('Subject no.')
    plt.ylabel('Mean')

    plt.show()

    new_model = LinearRegression().fit(meganewListx[40:2000, :], meganewListy1[40:2000].reshape((-1, 1)))
    r_sq = new_model.score(meganewListx[2000:, :], meganewListy1[2000:])
    print("coefficient of determination/r2 score for response time: ", r_sq)
    print("intercept: {new_model.intercept_}", new_model.intercept_)
    print("slope: {new_model.coef_}", new_model.coef_)
    y_pred = new_model.predict(meganewListx[2000:, :])
    #print('Accuracy = ', metrics.accuracy_score(np.round(meganewListy1[2000:]), np.round(numpy.asarray(y_pred))))
    coef = np.array([new_model.coef_[0][0]])
    coef1 = np.array([new_model.coef_[0][1]])

    intercept = np.array([new_model.intercept_])

    plt.figure()
    plt.scatter((coef * meganewListx[2000:, :1] + intercept).astype(float) + (coef1 * meganewListx[2000:, 1:2] + intercept).astype(float), meganewListy1[2000:].astype(float), color="deeppink")
    plt.plot((coef * meganewListx[2000:, :1] + intercept).astype(float) + (coef1 * meganewListx[2000:, 1:2] + intercept).astype(float), (numpy.asarray(y_pred)).astype(float), color="blue", linewidth=3)
    plt.title('Features vs Response Time')
    #plt.legend(['Actual Labels: Features on response time ', 'Predicted Labels: Features on response time'])
    plt.xlabel('Features: Latency and Peak Pupil Response')
    plt.ylabel('Response Time')
    plt.show()

    y_pred = new_model.predict(lstx)
    print('y_predictions for response time = ', y_pred)

    new_model = LinearRegression().fit(meganewListx[40:2000, :], meganewListy2[40:2000].reshape((-1, 1)))
    r_sq = new_model.score(meganewListx[2000:, :], meganewListy2[2000:])
    print("coefficient of determination/r2 score for hard/easy: ", r_sq)
    print("intercept: {new_model.intercept_}", new_model.intercept_)
    print("slope: {new_model.coef_}", new_model.coef_)
    y_pred = new_model.predict(meganewListx[2000:, :])
    #print('Accuracy = ', metrics.accuracy_score(meganewListy2[2000:], np.round(numpy.asarray(y_pred))))
    coef = np.array([new_model.coef_[0][0]])
    coef1 = np.array([new_model.coef_[0][1]])
    intercept = np.array([new_model.intercept_])

    plt.figure()
    plt.scatter((coef * meganewListx[2000:, :1] + intercept).astype(float) + (coef1 * meganewListx[2000:, 1:2] + intercept).astype(float), (meganewListy2[2000:]).astype(float), color="deeppink")
    plt.plot((coef * meganewListx[2000:, :1] + intercept).astype(float) + (coef1 * meganewListx[2000:, 1:2] + intercept).astype(float), (numpy.asarray(y_pred)).astype(float), color="blue", linewidth=3)
    plt.title('Features vs Hard/Easy')
    #plt.legend(['Actual Labels: Features on hard/easy ', 'Predicted Labels: Features on hard/easy'])
    plt.xlabel('Features: Latency and Peak Pupil Response')
    plt.ylabel('Hard/Easy')
    plt.show()

    y_pred = new_model.predict(lstx)
    print('y_predictions for hard/easy = \n', y_pred)

    new_model = LinearRegression().fit(meganewListx[40:2000, :], meganewListy3[40:2000].reshape((-1, 1)))
    r_sq = new_model.score(meganewListx[2000:, :], meganewListy3[2000:])
    print("coefficient of determination/r2 score for incorrect/correct: ", r_sq)
    print("intercept: {new_model.intercept_}", new_model.intercept_)
    print("slope: {new_model.coef_}", new_model.coef_)
    y_pred = new_model.predict(meganewListx[2000:, :])
    #print('Accuracy = ', metrics.accuracy_score(meganewListy3[2000:], np.round(numpy.asarray(y_pred))))
    coef = np.array([new_model.coef_[0][0]])
    coef1 = np.array([new_model.coef_[0][1]])
    intercept = np.array([new_model.intercept_])

    plt.figure()
    plt.scatter((coef * meganewListx[2000:, :1] + intercept).astype(float) + (coef1 * meganewListx[2000:, 1:2] + intercept).astype(float), (meganewListy3[2000:]).astype(float), color="deeppink")
    plt.plot((coef * meganewListx[2000:, :1] + intercept).astype(float) + (coef1 * meganewListx[2000:, 1:2] + intercept).astype(float), (numpy.asarray(y_pred)).astype(float), color="blue", linewidth=3)
    plt.title('Features vs Incorrect/Correct')
    #plt.legend(['Actual Labels: features on Incorrect/Correct ','Predicted Labels: features on Incorrect/Correct'])
    plt.xlabel('Features: Latency and Peak Pupil Response')
    plt.ylabel('Incorrect/Correct')
    plt.show()

    y_pred1 = new_model.predict(lstx)
    print('y_predictions for incorrect/correct = \n', y_pred1)

if __name__ == "__main__":

    main()








