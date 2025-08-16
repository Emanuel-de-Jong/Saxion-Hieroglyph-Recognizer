from flask import Flask, json, request
from flask_cors import CORS
import uuid, os

from classes.Model import Model

model_svm = Model()

api = Flask(__name__, static_url_path='')
# Enable CORS for this application (accessible from all domains)
CORS(api)

@api.route('/')
def root():
  return api.send_static_file('index.html')

# Please implement the route below:
@api.route('/classify', methods=['GET', 'POST'])
def classify_sample():
  # Get the uploaded image and save to temporary file
  image = request.files.get('image')
  temp_file = 'temp/{}.png'.format(uuid.uuid4())
  image.save(temp_file)

  ########################################################################
  ## Your logic here (please use classes to keep the server.py file clean)
  ## Path of file is in variable `temp_file
  ########################################################################
  
  label = model_svm.predict(temp_file)
  if label is None:
    label = 'empty'
    accuracy = 0.0
  else:
    accuracy = model_svm.get_accuracy()
  
  # Delete the temporary file
  if os.path.exists(temp_file):
    os.remove(temp_file)

  # Return JSON object with label and accuracy
  return json.dumps({"label": label, "accuracy": accuracy})

if __name__ == '__main__':
  api.run()