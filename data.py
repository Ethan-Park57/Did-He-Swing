import pandas as pd
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
from pybaseball import cache
import numpy as np
import unidecode
import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# This function is a scratchpad for testing code
def scratchpad():
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.max_rows', None)
    #     raw_family_name = 'Urquidy'
    #     raw_first_name = 'Jose'
    #     raw_first_name = unidecode.unidecode(raw_first_name)
    #     raw_family_name = unidecode.unidecode(raw_family_name)
    #     x = playerid_lookup(raw_family_name, raw_first_name)['key_mlbam'].astype(int)
    #     sanmartin_id = list(x)[0]
    #     print(sanmartin_id)

    # sanmartin_stats = statcast_pitcher('2021-10-03', '2021-10-03', sanmartin_id)
    # print(pd.unique(sanmartin_stats['events']))
    # print(pd.unique(sanmartin_stats['description']))
    # print(pd.unique(sanmartin_stats['balls']))
    # print(pd.unique(sanmartin_stats['strikes']))

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



# This function gets the data necessary to perform a KNN analysis into a .csv file
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


if __name__ == '__main__':
    # data_wrangling()
    scratchpad()
