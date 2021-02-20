import json
import requests
from fpdf import FPDF
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.dates as mdates
import locale
import math


def gauss(x, mu, sigma, scale):
    return scale * np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2)))


def weib(x, k, a, b, g):
    return k * g * b * (a ** b) * np.exp(-1 * g * ((a / x) ** b)) / (x ** (b + 1))


def fit(func, x, y, start):

    outliersweight = None
    for i in range(100):
        popt, pcov = curve_fit(
            func, x, y, start, sigma=outliersweight, maxfev=10000)
        pred = np.array([func(px, *popt) for px in x])
        outliersweight = np.abs(pred - y)
        outliersweight = 1 - np.tanh(outliersweight)
        outliersweight = outliersweight / np.max(outliersweight)
        outliersweight = sp.special.softmax(1 - outliersweight)
    return func(x, *popt), popt


def daily_last14_deaths_plot(pdf):
    deaths = greece_data[['new_deaths', 'date']]
    deaths = deaths.dropna().reset_index().drop('index', axis=1)
    dates = pd.to_datetime(deaths.date)
    days = dates[-14:]
    l = len(dates)
    labels = list()
    for i in range(14):
        labels.append(days[l-1-i].strftime("%d-%B-%Y"))
    plt.figure()
    plt.plot([i for i in range(14)], deaths.new_deaths[-14:],
             '-o', label=f"Μέσος όρος %.3f" % np.mean(deaths.new_deaths[-14:]))
    plt.title("Θανάτοι των τελευταίων 14 ημερών")
    plt.xticks([i for i in range(0, 14)],
               labels[::-1], rotation=60)
    plt.ylabel("Θανάτοι")
    plt.tight_layout()
    plt.legend()
    plt.savefig("images/last_14_days_deaths.png")

    plt.figure()

    fig = plt.figure()

    dates = pd.to_datetime(deaths.date)
    months = [dates[i] for i in range(0, len(deaths.date), 30)]
    labels = list()
    for i in range(len(months)):
        labels.append(months[i].strftime("%B-%Y"))
    plt.bar([i for i in range(len(deaths.new_deaths))], deaths.new_deaths)

    y, popt_weib_daily = fit(func=weib, x=[i for i in range(
        1, len(deaths.new_deaths) + 1, 1)], y=deaths.new_deaths, start=[1000, 14, 4, 500])

    plt.plot([i for i in range(1, len(deaths.new_deaths) + 1, 1)],
             y, color='y', label="Weibull")

    y, popt_gauss_daily = fit(func=gauss, x=[i for i in range(
        1, len(deaths.new_deaths) + 1, 1)], y=deaths.new_deaths, start=[0, 20, 100])

    plt.plot([i for i in range(1, len(deaths.new_deaths) + 1, 1)],
             y, color='r', label="Gaussian")

    plt.title("Κατανομή ημερίσιων θανάτων")
    plt.xticks([i for i in range(0, len(deaths.date), 30)],
               labels, rotation=60)
    plt.ylabel("Θάνατοι")
    plt.tight_layout()

    plt.savefig("images/deaths_daily.png")
    epw = pdf.w - 2*pdf.l_margin
    col_width = epw/4
    th = pdf.font_size
    pdf.add_page()

    pdf.set_font('DejaVu', '', 8)

    pdf.cell(
        epw, 0.0, 'θανάτοι των τελευταίων 14 ημερών')

    pdf.ln(th)

    pdf.cell(col_width, th, str(
        "Ημερομηνία"), border=1, align='C')

    pdf.cell(col_width, th, str(
        "Θανάτοι"), border=1, align='C')

    pdf.ln(th)

    l = len(deaths)
    pdf.set_font('DejaVu', '', 8)

    for i in range(14):
        pdf.cell(col_width, th, str(
            deaths.iloc[l-i-1, 1]), border=1, align='C')
        pdf.cell(col_width, th, str(
            deaths.iloc[l-i-1, 0]), border=1, align='C')
        pdf.ln(th)

    pdf.image("images/deaths_daily.png", x=0, y=60, w=100, h=90)
    pdf.image("images/last_14_days_deaths.png", x=105, y=60, w=100, h=90)

    return deaths


def cases_daily_plot(cases, pdf):
    encoder = LabelEncoder()

    x = encoder.fit_transform(cases[['date']]) + 1

    fig = plt.figure()

    dates = pd.to_datetime(cases.date)
    months = [dates[i] for i in range(0, len(cases.date), 30)]
    labels = list()
    for i in range(len(months)):
        labels.append(months[i].strftime("%B-%Y"))

    plt.bar(x, cases.new_cases)

    y, _ = fit(func=weib, x=x,
               y=cases.new_cases, start=[30000, 14, 4, 500])
    plt.plot(x, y*1.2, color='y', label="Weibull")

    y, _ = fit(func=gauss, x=x,
               y=cases.new_cases, start=[0, 20, 100])
    plt.plot(x, y*1.2, color='r', label="Gaussian")
    plt.title("Κατανομή ημερίσιων κρουσμάτων")
    plt.ylabel('Κρούσματα')
    plt.xticks([i for i in range(0, len(cases.date), 30)],
               labels, rotation=60)
    plt.tight_layout()

    plt.savefig("images/cases_daily")

    epw = pdf.w - 2*pdf.l_margin
    col_width = epw/4
    th = pdf.font_size

    pdf.set_font('DejaVu', '', 8)

    pdf.cell(
        epw, 0.0, 'Αριθμός κρουσμάτων τις τελευταίες 14 ημέρες')

    pdf.ln(th)

    pdf.cell(col_width, th, str(
        "Ημερομηνία"), border=1, align='C')

    pdf.cell(col_width, th, str(
        "Κρούσματα"), border=1, align='C')

    pdf.ln(th)

    l = len(cases)
    pdf.set_font('DejaVu', '', 8)

    for i in range(14):
        pdf.cell(col_width, th, str(
            f'{dates[l-i-1].date().year}-{str(dates[l-i-1].date().month).zfill(2)}-{str(dates[l-i-1].date().day).zfill(2)}'), border=1, align='C')
        pdf.cell(col_width, th, str(
            cases.iloc[l-i-1, 1]), border=1, align='C')
        pdf.ln(th)

    pdf.image("images/cases_daily.png", x=0, y=120, w=100, h=90)
    return cases


def case_test_perc_plot(pdf):

    fig = plt.figure()
    c = greece_data[["new_tests", "date"]]
    c = c.groupby('date')['new_tests'].sum().reset_index()
    c = c[c.new_tests > 0]
    indexes = c.index
    new_cases_tests = greece_data[['date', 'new_cases']
                                  ].reset_index().dropna().drop('index', axis=1)
    new_cases_tests = new_cases_tests.loc[indexes]
    new_cases_tests = pd.concat((new_cases_tests, c.new_tests), axis=1)
    new_cases_tests['percent'] = (
        new_cases_tests.new_cases / new_cases_tests.new_tests) * 100
    new_cases_tests = new_cases_tests.reset_index().drop('index', axis=1)

    dates = pd.to_datetime(new_cases_tests.date)
    months = [dates[i] for i in range(0, len(new_cases_tests.date), 30)]
    labels = list()
    for i in range(len(months)):
        labels.append(months[i].strftime("%B-%Y"))

    plt.scatter([i for i in range(len(new_cases_tests))],
                new_cases_tests.percent / 100.0, s=5)

    plt.bar([i for i in range(len(new_cases_tests))],
            new_cases_tests.percent / 100.0)
    plt.xticks([i for i in range(0, len(new_cases_tests.date), 30)],
               labels, rotation=60)

    plt.title("Κρούσματα / Τεστ")
    plt.ylabel("Ποσοστό θετικότητας")
    plt.tight_layout()

    plt.savefig("images/case_test.png")

    epw = pdf.w - 2*pdf.l_margin
    col_width = epw/4
    th = pdf.font_size

    pdf.cell(
        epw, 0.0, 'Ποσοστό θετικότητας', align='C')

    pdf.ln(th)

    pdf.cell(col_width, th, str(
        "Ημερομηνία"), border=1, align='C')

    pdf.cell(col_width, th, str(
        "Κρούσματα"), border=1, align='C')

    pdf.cell(col_width, th, str(
        "Τεστ"), border=1, align='C')

    pdf.cell(col_width, th, str(
        "Ποσοστό"), border=1, align='C')

    l = len(new_cases_tests)
    pdf.set_font('DejaVu', '', 8)
    for i in range(14):
        pdf.cell(col_width, th, str(
            f'{dates[l-i-1].date().year}-{str(dates[l-i-1].date().month).zfill(2)}-{str(dates[l-i-1].date().day).zfill(2)}'), border=1, align='C')
        pdf.cell(col_width, th, str(
            new_cases_tests.iloc[l-i-1, 1]), border=1, align='C')
        pdf.cell(col_width, th, str(
            new_cases_tests.iloc[l-i-1, 2]), border=1, align='C')
        pdf.cell(col_width, th, str(
            new_cases_tests.iloc[l-i-1, 3]), border=1, align='C')
        pdf.ln(th)

    pdf.ln(10)
    pdf.image("images/case_test.png",  x=100, y=120, w=100, h=90)
    return new_cases_tests


def cases_weekly_plot(cases_week, pdf):

    fig = plt.figure()

    x = [i+1 for i in range(len(cases_week.date))]

    dates = cases_week.date
    months = [dates[i] for i in range(0, len(cases_week.date), 4)]
    labels = list()
    for i in range(len(months)):
        labels.append(months[i].strftime("%B-%Y"))

    plt.bar(x, cases_week.new_cases)

    y, popt_weib_week = fit(func=weib, x=x,
                            y=cases_week.new_cases, start=[30000, 14, 4, 500])
    plt.plot(x, y, color='y', label="Weibull")

    y, popt_gauss_week = fit(func=gauss, x=x,
                             y=cases_week.new_cases, start=[0, 20, 100])
    plt.plot(x, y, color='r', label="Gaussian")

    plt.title("Κατανομή εβδομαδιαίων κρουσμάτων")
    plt.xticks([i for i in range(0, len(cases_week.date), 4)],
               labels, rotation=60)
    plt.ylabel("Κρούσματα")
    plt.tight_layout()

    plt.savefig("images/cases_weekly")
    pdf.image("images/cases_weekly.png", x=0, y=207, w=100, h=90)

    return cases_week


def model(cases, pdf):
    scaler = MinMaxScaler()

    cases['new_cases'] = scaler.fit_transform(cases[['new_cases']])

    xtrain = []
    ytrain = []
    for i in range(14, len(cases.new_cases)):
        xtrain.append(cases.new_cases[i-14:i])
        ytrain.append(cases.new_cases[i])
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)

    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)

    def prepare_tf():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        return tf.compat.v1.Session(config=config)

    prepare_tf()

    nn = Sequential()

    nn.add(LSTM(units=50, return_sequences=True,
                input_shape=(xtrain.shape[1], 1)))
    nn.add(Dropout(0.2))

    nn.add(LSTM(units=50, return_sequences=True))
    nn.add(Dropout(0.2))

    nn.add(LSTM(units=50, return_sequences=True))
    nn.add(Dropout(0.2))

    nn.add(LSTM(units=50))
    nn.add(Dropout(0.2))

    nn.add(Dense(units=1))

    nn.compile(optimizer="adam", loss="mean_squared_error")

    try:
        nn.load_weights("nn.h5")
    except OSError:
        nn.fit(xtrain, ytrain, epochs=100, batch_size=32)
        nn.save_weights('nn.h5')

    l = len(cases.new_cases)
    xtest = np.array(cases.new_cases[l-14:l])
    xtest = xtest.reshape(1, 14, 1)

    new_cases_prediction_sc = nn.predict(xtest)
    new_cases_prediction = scaler.inverse_transform(new_cases_prediction_sc)

    pdf.set_font('DejaVu', '', 16)

    pdf.write(230, "Πρόβλεψη για " +
              datetime.today().strftime('%Y-%m-%d') + " :" + str(math.floor(new_cases_prediction[0][0])))


def last_14_days_cases_plot(cases, pdf):
    dates = pd.to_datetime(cases.date)
    days = dates[-14:]
    l = len(dates)
    labels = list()
    for i in range(14):
        labels.append(days[l-1-i].strftime("%d-%B-%Y"))

    plt.figure()
    plt.plot([i for i in range(14)],
             cases.new_cases[-14:], '-o', label=f"Μέσος όρος %.3f" % np.mean(cases.new_cases[-14:]))
    plt.title("Κρούσματα των τελευταίων 14 ημερών")
    plt.xticks([i for i in range(0, 14)],
               labels[::-1], rotation=60)
    plt.ylabel("Κρούσματα")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/last_14_days_cases.png")

    epw = pdf.w - 2*pdf.l_margin
    col_width = epw/4
    pdf.image("images/last_14_days_cases.png", x=100, y=207, w=100, h=90)


def prepare_pdf():
    pdf = FPDF()
    pdf.add_page()
    epw = pdf.w - 2*pdf.l_margin
    pdf.add_font('DejaVu', '', 'DejaVuLGCSans-BoldOblique.ttf', uni=True)
    pdf.set_font('DejaVu', '', 24)
    pdf.ln(60)
    pdf.cell(
        epw, 0.0, 'Covid Analytics Report', align='C')
    pdf.ln(10)
    pdf.set_font('DejaVu', '', 16)
    pdf.cell(
        epw, 0.0, datetime.today().strftime("%Y-%m-%d"), align='C')
    pdf.ln(5)

    pdf.add_page()

    pdf.set_font('DejaVu', '', 8)

    return pdf


def age_distribution_plot(data, title, ylabel, file, pdf, x, y):
    data.date = pd.to_datetime(data.date)
    encoder = LabelEncoder()
    X = encoder.fit_transform(data[['date']]) + 1

    fig = plt.figure()

    dates = pd.to_datetime(data.date)
    months = [dates[i] for i in range(0, len(data.date), 30)]
    labels = list()
    for i in range(len(months)):
        labels.append(months[i].strftime("%B-%Y"))

    plt.plot(X, data['0-17'], color='r', label="0-17")
    plt.plot(X, data['18-39'], color='g', label="18-39")
    plt.plot(X, data['40-64'], color='b', label='40-64')
    plt.plot(X, data['65+'], color='y', label='65+')

    plt.xticks([i for i in range(0, len(data.date), 30)],
               labels, rotation=60)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    plt.savefig(file)

    pdf.image(file, x=x, y=y, w=100, h=90)


def create_report(case, cases_week, age_dist_cases, age_dist_deaths):

    pdf = prepare_pdf()

    case_test_perc_plot(pdf)
    cases_daily_plot(case, pdf)
    cases_weekly_plot(cases_week, pdf)
    last_14_days_cases_plot(case, pdf)
    daily_last14_deaths_plot(pdf)
    age_distribution_plot(age_dist_cases, title="Συνολικά κρούσματα ανα ηλικία",
                          ylabel="Κρούσματα", file="images/age_cases.png", pdf=pdf, x=0, y=155)

    age_distribution_plot(age_dist_deaths, title="Συνολικοί θανάτοι ανα ηλικία",
                          ylabel="Θανάτοι", file="images/age_deaths.png", pdf=pdf, x=105, y=155)
    model(case, pdf)

    pdf.output("report.pdf", "F")


data = pd.read_csv(
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")


greece_data = data[data['location'] == 'Greece']

cases = greece_data[["new_cases", "date"]]
cases = cases.dropna(axis='rows')
cases = cases.reset_index(drop=True)
cases = cases.groupby('date')['new_cases'].sum().reset_index()


cases_week = greece_data[["new_cases", "date"]]
cases_week.date = pd.to_datetime(
    cases_week['date'], errors='coerce', format='%Y-%m-%d')
cases_week.index = cases_week['date']
cases_week = cases_week.drop('date', axis=1)

cases_week = cases_week.resample('W').sum()

cases_week = cases_week.reset_index()


d = requests.get(
    "https://covid-19-greece.herokuapp.com/age-distribution-history")
data_list = json.loads(d.text)['age-distribution']
date = list()
cases1 = list()
cases2 = list()
cases3 = list()
cases4 = list()
for row in data_list:
    date.append(row['date'])
    cases1.append(row['cases']['0-17'])
    cases2.append(row['cases']['18-39'])
    cases3.append(row['cases']['40-64'])
    cases4.append(row['cases']['65+'])
data_dict = {'date': date, '0-17': cases1,
             '18-39': cases2, '40-64': cases3, '65+': cases4}
age_dist_cases = pd.DataFrame(data=data_dict)

date = list()
cases1 = list()
cases2 = list()
cases3 = list()
cases4 = list()
for row in data_list:
    date.append(row['date'])
    cases1.append(row['deaths']['0-17'])
    cases2.append(row['deaths']['18-39'])
    cases3.append(row['deaths']['40-64'])
    cases4.append(row['deaths']['65+'])
data_dict = {'date': date, '0-17': cases1,
             '18-39': cases2, '40-64': cases3, '65+': cases4}
age_dist_deaths = pd.DataFrame(data=data_dict)

create_report(cases, cases_week, age_dist_cases, age_dist_deaths)
