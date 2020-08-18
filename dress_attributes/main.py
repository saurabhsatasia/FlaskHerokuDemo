# doing necessary imports

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import requests
from urllib.request import urlopen as uReq
from pymongo import MongoClient
from datetime import datetime

import pickle

from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)  # initializing a flask app

client = MongoClient('localhost', 27017)  # for running locally
# client = MongoClient(os.environ['DB_PORT_27017_TCP_ADDR'], 27017) # ip and port, for local checking it would be 'localhost' only
db = client['Sales']


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict/', methods=['GET'])  # route to display the home page
@cross_origin()
def predictionPage():
    return render_template("prediction.html")


@app.route('/predict_button', methods=['POST'])  # route to display the home page
@cross_origin()
def resultPage():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            # dress_id = int(request.form['dress_id'])
            Style = request.form['Style']
            Price = request.form['Price']
            Rating = int(request.form['Rating'])
            Size = request.form['Size']
            Season = request.form['Season']
            NeckLine = request.form['NeckLine']
            SleeveLength = request.form['SleeveLength']
            waiseline = request.form['Waiseline']
            Material = request.form['Material']
            PatternType = request.form['PatternType']
            total_sale = int(request.form['total_sale'])

            userInput = [
                [Style, Price, Rating, Size, Season, NeckLine, SleeveLength, waiseline, Material, PatternType,
                 total_sale]]
            filename = 'dressRecon_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

            # predictions using the loaded model file
            onehotencoder = pickle.load(
                open('onehotencoder_model.pickle', 'rb'))  # need to use as training model is onehot encoded
            prediction = loaded_model.predict(onehotencoder.transform(userInput))
            print('prediction is', prediction)

            # showing the prediction results in a UI
            if prediction[0] == 1:
                prediction1 = 'Yes'
            else:
                prediction1 = 'No'

            # insert into mongoDB

            dbCollection = db['PredictionStored']
            record1 = {"Style": Style, "Price": Price, "ProductRating": str(Rating), "Size": Size,
                       "Season": Season, "Neckline": NeckLine, "SleeveLength": SleeveLength, "Waiseline": waiseline,
                       "Material": Material, "PatternType": PatternType, "ProductCount": str(total_sale),
                       "BuyIt ?": prediction1, "DateTime": datetime.now()}
            recordInserted = dbCollection.insert_one(record1)
            # print(recordInserted)

            return render_template('results.html', prediction=prediction1)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
        # return render_template('results.html')

    else:
        return render_template('index.html')


@app.route('/contacts/', methods=['GET'])  # route to display the home page
@cross_origin()
def contactPage():
    return render_template("contacts.html")


@app.route('/about/')  # route to display the home page
@cross_origin()
def aboutPage():
    return render_template("about.html")


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True) # running the app
    # app.run(port=8000, debug=True)  # running the app on the local machine on port 8000
