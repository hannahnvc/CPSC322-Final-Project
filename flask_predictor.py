import requests # lib for making requests
import json # lib for parsing strings/JSON objects

# url = "https://movie-forest-flask-app.herokuapp.com/predict?" add stuff after predict 
url = "http://0.0.0.0:5000/predict?imdb_title_id=268978&title=A+Beautiful+Mind&original_title=A+Beautiful+Mind&year=2001.0&date_published=2002-02-22&genre=Biography+Drama&duration=135.0&country=USA&language=English&writer=Ron+Howard+Akiva+Goldsman+Sylvia+Nasar&production_company=Universal+Pictures&actors=Russell+Crowe+Ed+Harris+Jennifer+Connelly+Christopher+Plummer+Paul+Bettany+Adam+Goldberg+Josh+Lucas+Anthony+Rapp+Jason+Gray-Stanford+Judd+Hirsch+Austin+Pendleton+Vivien+Cardone+Jillie+Simon+Victor+Steinbach+Tanya+Clarke&avg_vote=8.2&votes=827595.0&reviews_from_users=11&reviews_from_critics=2"
# make the GET request
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
response = requests.get(url=url)

# first, check the status code!!
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
status_code = response.status_code
print("status_code:", status_code)
print("message body:", response.text)

if status_code == 200:
    # success! can grab message body 
    json_object = json.loads(response.text)
    print(json_object)