from flask import Flask, redirect, render_template, request, session, url_for
import pandas as pd
from pybaseball import statcast_pitcher
import unidecode
import datetime
import matplotlib
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = Flask(__name__)

# root endpoint
@app.route("/")
def home():
    return render_template('home.html')

# page with form to submit features for prediction
@app.route("/predict")
def predict():
    return render_template('predict.html')

# form to submit features for prediction
@app.route("/action/predict", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        # get submitted parameters
        inning = request.form['inning']
        wOBA = request.form['wOBA']
        outs = request.form['outs']

        # Make string -> int, float, int?
        inning = int(inning)
        wOBA = float(wOBA)
        outs = int(outs)

        # get the prediction from the model
        prediction = make_prediction(instantiate_model(), inning, wOBA, outs)

        # serve html w/ prediction
        return render_template('prediction.html', prediction=prediction)


# Displays existing data about first pitch out swings
@app.route("/display")
def display():
    # create plot and return url
    plot_url = create_plot()
    # get total swings and non-swings
    freqs = sum_results()
    print(freqs[0.0])
    # return html with bar plot url
    return render_template('display.html', plot_url=plot_url, swings=freqs[1.0], non_swings=freqs[0.0])

# Creates the trained KNN model based on past data
def instantiate_model():
    df = pd.read_csv('data.csv')
    df.dropna(inplace=True)

    # get features and target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # scale features
    myScaler = StandardScaler()
    myScaler.fit(X)
    X_scaled = pd.DataFrame(myScaler.transform(X), columns=X.columns)

    # split into training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
    # use n_neighbors = 5 from previous testing, which can be seen below.
    model = KNeighborsClassifier(n_neighbors=5)
    # train model
    model.fit(X_train, y_train)

    # return trained model
    return model

# Using inning=6, wOBA=0.5, outs=1 yields 'Swing!'
# Takes the user submitted feature values and makes a prediction about whether the next pitch will be a swing
def make_prediction(model, inning, wOBA, outs):
    # create sample and make prediction
    sample = [[inning, wOBA, outs]]
    sample_pred = model.predict(sample)

    # Return the prediction
    print(sample_pred)
    if sample_pred[0] == 0:
        return 'Non-Swing!'
    else:
        return 'Swing!'


# Aggregate past data about swings
def sum_results():
    df = pd.read_csv('data.csv')
    print(df)
    return df['Swing'].value_counts().to_dict()


# Create the plot to be display for EXISTING data
def create_plot():
    # Create the bar plot
    img = BytesIO()
    df = pd.read_csv('data.csv')
    swing = df['Swing']
    swing.dropna(inplace=True)
    swing.value_counts().plot.bar()
    plt.title('Frequencies of swings vs non-swings after first pitch out')
    plt.ylabel('Frequency')
    plt.xlabel(['0.0 = Non-Swing', '1.0 = Swing'])
    plt.legend()

    # Save the bar plot
    plt.savefig(img, format='png')
    plt.close()

    # Get the url for the bar plot
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url

# This function shows the code used to find the best k value to use for KNN.
# After running the model with several values of k, 5 was a good middle ground.
# This code is never run during the app's lifecycle, it's here for display only
def model_selection():
    df = pd.read_csv('data.csv')

    df.dropna(inplace=True)

    myScaler = StandardScaler()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    myScaler.fit(X)
    X_scaled = pd.DataFrame(myScaler.transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
    X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)
    trainA_accuracy = []
    trainB_accuracy = []

    neighbors = np.arange(1, 50)
    for k in neighbors:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_predA = model.predict(X_trainA)
        y_predB = model.predict(X_trainB)
        trainA_accuracy.append(metrics.accuracy_score(y_trainA, y_predA))
        trainB_accuracy.append(metrics.accuracy_score(y_trainB, y_predB))

    plt.plot(neighbors, trainA_accuracy, '--r', label='Train A')
    plt.plot(neighbors, trainB_accuracy, label='Train B')
    plt.xlabel('# of neighbors (k)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(neighbors)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    print('\n\n\nConfusion matrix for the test set:', metrics.confusion_matrix(y_test, y_pred_test))
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    print(metrics.accuracy_score(y_test, y_pred_test))
    plt.show()


# This function gets the data necessary to perform a KNN analysis into a .csv file.
# It is never run in the app, but I wanted to show the process for how the .csv files I used are created.
def data_wrangling():
    # read the df
    raw_df = pd.read_csv('savant_data_5000.csv')

    # limit to these cols: 'inning', 'estimated_woba_using_speedangle', 'outs_when_up', 'player_name', 'game_date',
    # 'at_bat_number', 'pitcher'
    current_pitch = raw_df[['inning', 'estimated_woba_using_speedangle', 'outs_when_up',
                            'player_name', 'game_date', 'at_bat_number', 'pitcher']]
    next_pitch_df = pd.DataFrame()

    # create a new df with 'inning', 'estimated_woba_using_speedangle', 'outs_when_up' which we will use for KNN
    my_df = current_pitch.loc[:, ['inning', 'estimated_woba_using_speedangle', 'outs_when_up']]

    # iterate thru df
    for i in range(len(current_pitch.index)):
        # store pertinent data
        raw_first_name = unidecode.unidecode(current_pitch.loc[i, 'player_name'].split(', ')[1])
        raw_family_name = unidecode.unidecode(current_pitch.loc[i, 'player_name'].split(', ')[0])
        date = current_pitch.loc[i, 'game_date']
        date = datetime.datetime.strptime(date, '%m/%d/%y').strftime('%Y-%m-%d')
        inning = current_pitch.loc[i, 'inning']

        # get pitcher's stats from that game
        print(raw_first_name, raw_family_name)
        retrieved_game_stats = statcast_pitcher(date, date, current_pitch.loc[i, 'pitcher'])

        # get the at_bat_number from the df
        at_bat_number = current_pitch.loc[i, 'at_bat_number']
        # if at_bat_number + 1 exists in the pitcher's stats returned from statcast_pitcher
        if at_bat_number + 1 in retrieved_game_stats['at_bat_number'].values:
            next_pitch_series = retrieved_game_stats.loc[retrieved_game_stats['at_bat_number'] == at_bat_number + 1,
                                                         ['inning', 'description']].iloc[-1]
            next_pitch_df = pd.concat([next_pitch_df, next_pitch_series.to_frame().T], ignore_index=True)
            # check that the inning is the same inning from the df
            if next_pitch_series['inning'] == inning:
                # if first pitch is a 'hit_into_play' 'swinging_strike' 'foul' in description
                # then add a 'swing' (1) to my df
                if next_pitch_series['description'] == 'hit_into_play' or \
                        next_pitch_series['description'] == 'swinging_strike' or \
                        next_pitch_series['description'] == 'foul':
                    my_df.loc[my_df.index[i], 'Swing'] = 1
                # else if first pitch is a 'ball' 'called_strike' 'blocked_ball' in description
                # then add a 'take' (0) to my df
                else:
                    my_df.loc[my_df.index[i], 'Swing'] = 0

    my_df.to_csv('data.csv', index=False)
    next_pitch_df.to_csv('next_pitch_result.csv', index=False)


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)