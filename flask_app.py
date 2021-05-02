# we are going to use a micro web framework called Flask
# to create our web app (for running an API service)
import os 
from flask import Flask, jsonify, request 
import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyNaiveBayesClassifier
import mysklearn.myevaluation
import mysklearn.myevaluation as myevaluation
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import copy

# by default flask runs on port 5000

app = Flask(__name__)

# we need to define "routes", functions that 
# handle requests
# let's add a route for the "homepage"
@app.route("/", methods=["GET"])
def index():
    # return content and status code
    return "Our Naive Classifier was best for this Datset. It was the only classifier to predict both Winners and Nominees Correctly.", 200

def naive_bayes_predict(table, instance):
    NaiveBayes = MyNaiveBayesClassifier()
    X = []
    y = []
    data_table = copy.deepcopy(table)
    data_table.remove_rows_with_missing_values()
    data = copy.deepcopy(data_table.data)
    for row in data:
        new_row = []
        y.append(row[-1])
        for item in row[:17]:
            new_row.append(item)
        X.append(new_row)
    X_half = []
    y_half = []
    for i in range(int((len(X)/2)), (len(X))):
        X_half.append(X[i])
        y_half.append(y[i])
    X_half_edit = []
    for i in range(len(X_half)):
        new_row = []
        for j in range(len(X_half[0])):
            if j != 0:
                if j!= 1:
                    if j!= 2:
                        if j!= 3:
                            if j!= 4:
                                if j!= 5:  
                                    if j!= 9:
                                        if j!= 10:
                                            if j!= 11:
                                                if j!= 12:
                                                    new_row.append(X_half[i][j])
        X_half_edit.append(new_row)
    X_train, X_test, y_train, y_test = myevaluation.train_test_split(X_half_edit, y_half)
    NaiveBayes.fit(X_train, y_train)
    predicted = NaiveBayes.predict([instance])
    print(predicted)
    return predicted

def predict_interviews_well(instance):
    pytable = MyPyTable()
    pytable.load_from_file("film_data.csv")
    print("col_names", pytable.column_names)
    print("instance1")
    print("instance", instance)
    try:
        return naive_bayes_predict(pytable.data, instance)
    except:
        # something went wrong
        return None 

# add a route for "/predict" API endpoint
@app.route("/predict", methods=["GET"])
def predict():
    # goal is to parse the query string to the args
    # query args are in the request object
    imdb_title_id = request.args.get("imdb_title_id", "") # change these to relate to the movie dataset
    title = request.args.get("title", "")
    original_title = request.args.get("original_title", "")
    year = request.args.get("year", "")
    date_published = request.args.get("date_published", "")
    genre = request.args.get("genre", "")
    duration = request.args.get("duration", "")
    country = request.args.get("country", "")
    language = request.args.get("language", "")
    director = request.args.get("writer", "")
    writer = request.args.get("writer", "")
    production_company = request.args.get("production_company", "")
    actors = request.args.get("actors", "")
    avg_vote = request.args.get("avg_vote", "")
    votes = request.args.get("votes", "")
    reviews_from_users = request.args.get("reviews_from_users", "")
    reviews_from_critics = request.args.get("reviews_from_critics", "")
    print("imdb_title_id:", imdb_title_id,title,original_title,year,date_published,genre,duration,country,language,director,writer,production_company,actors,avg_vote,votes,reviews_from_users,reviews_from_critics)
    # task: extract the other three parameters
    # level, lang, tweets, phd
    # make a prediction with the tree
    # respond to the client with the prediction in a JSON object
    prediction = predict_interviews_well([imdb_title_id,title,original_title,year,date_published,genre,duration,country,language,director,writer,production_company,actors,avg_vote,votes,reviews_from_users,reviews_from_critics])
    if prediction is not None:
        result = {"prediction": prediction} 
        print(jsonify(result))
        return jsonify(result), 200 
    else:
        return "Error making prediction", 400



if __name__ == "__main__":
    # deployment notes
    # two main categories of deployment
    # host your own server OR use a cloud provider
    # quite a few cloud provider options... AWS, Heroku, Azure, DigitalOcean,...
    # we are going to use Heroku (backend as a service BaaS)
    # there are quite a few ways to deploy a Flask app on Heroku
    # 1. deploy the app directly on an ubuntu "stack" (e.g. Procfile and requirements.txt)
    # 2. deploy the app as a Docker container on a container "stack" (e.g. Dockerfile)
    # 2.A. build a Docker image locally and push the image to a container registry 
    # (e.g. Heroku's registry)
    # **2.B.** define a heroku.yml and push our source code to Heroku's git
    # and Heroku is going to build the Docker image (and register it)
    # 2.C. define main.yml and push our source code to Github and Github
    # (via a Github Action) builds the image and pushes the image to Heroku

    # we need to change app settings for deployment
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port)