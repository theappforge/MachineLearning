import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras import initializations
import pandas as pd
import pickle


def my_init(shape, name=None):
    return initializations.zero(shape, name=name)


def basemodel(num_input):
    model = Sequential()
    model.add(Dense(50, input_dim=num_input, init=my_init, activation='relu'))
    model.add(Dense(1, init='normal', activation='linear'))

    # Compile model     # loss     # method
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def find_derivative(df, model):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())

    w1 = weights[0][0]
    w2 = weights[0][1]
    derivative = list(np.dot(w1, w2))

    columns = df.columns.values

    f = open('Telkaderivative.csv', 'w')
    for i in range(0, len(derivative)):
        f.write('{},'.format(columns[i]))
    f.write("\n")
    for der in derivative:
        f.write('{},'.format(der))
    f.close()


def train_model(EPOCH, df):
    np.random.seed(0)

    X = np.array(df.drop('userMark', 1))
    y = np.array(df['userMark'])

    num_input = X.shape[1]

    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Fit the model
    net = basemodel(num_input)
    net.fit(X_train, y_train, nb_epoch=EPOCH, batch_size=8)

    # evaluate the model
    result = net.predict(X_test)

    print("MSE: ", mean_squared_error(y_test, result))

    scores = net.evaluate(X_test, y_test, verbose=1)
    print("\n%s: %.2f%%" % (net.metrics_names[1], scores[1] * 100))

    pickle.dump(net, open("Telkanet({}%)_{}.pickle".format(round(scores[1] * 100, 2), EPOCH), "wb"))

    return net


def main():
    np.random.seed(0)
    EPOCH = 200
    # data reading
    df = pd.read_csv("data/films.csv")

    for item in df.columns.values:
        if "kinooiskRating" not in item \
                and "imdbRating" not in item \
                and "imdbVotes" not in item \
                and "userMark" not in item \
                and "Metascore" not in item \
                and "Genre" not in item \
                and "CountCriticalReviews" not in item:
                # and "Year" not in item \

            df = df.drop(item, 1)

    df = df.drop([0])

    # df['poisk_imdb'] = np.sqrt(np.array(df["kinooiskRating"].astype(float)) + np.array(df["imdbRating"]).astype(float))

    # df = df.drop("kinooiskRating", 1).drop("imdbRating", 1)

    print(df.head())

    # str_float(df)

    net = train_model(EPOCH, df)

    # file = open("Vadimnet(92.86%)_200.pickle", 'rb')
    # net = pickle.load(file)

    # find_derivative(df, net)


if __name__ == "__main__":
    main()