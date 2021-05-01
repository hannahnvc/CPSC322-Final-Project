import requests # lib for making requests
import json # lib for parsing strings/JSON objects

# url = "https://movie-forest-flask-app.herokuapp.com/predict?" add stuff after predict 
url = "http://0.0.0.0:5000/predict?level=Junior&lang=Java&tweets=yes&phd=yes" # add stuff after predict

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