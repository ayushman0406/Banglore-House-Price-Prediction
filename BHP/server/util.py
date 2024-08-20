import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__last_four_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath, area_type):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    # Set area type in the last four columns
    for i, col in enumerate(__last_four_columns):
        if col == area_type:
            x[__data_columns.index(col)] = 1
        else:
            x[__data_columns.index(col)] = 0

    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __last_four_columns

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:-4]  # first 3 columns are sqft, bath, bhk
        __last_four_columns = __data_columns[-4:]  # last 4 columns for development time

    global __model
    if __model is None:
        with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_location_names():
    return __locations


def get_last_four_columns():
    return __last_four_columns


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_last_four_columns())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3, 'super built-up area'))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2, 'built-up area'))
    print(get_estimated_price('Kalhalli', 1000, 2, 2, 'plot area'))  # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2, 'carpet area'))  # other location
