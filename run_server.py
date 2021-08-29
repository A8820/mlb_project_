# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python request.py

import os

# import the necessary packages
import dill
import pandas as pd

dill._dill._reverse_typemap['ClassType'] = type
# import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


modelpath = "./models/catreg_pipeline.dill"
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        date, curs_y, WA_TOD_RATE, VALUE_TOD, WA_TOM_RATE, VALUE_TOM, Rate, value_x, value_y, Brent, t_1, t_2 = None, None, None, None, None, None, None, None, None, None, None, None

        request_json = flask.request.get_json()

        if request_json["date"]:
            date = request_json['date']

        if request_json["curs_y"]:
            curs_y = request_json['curs_y']

        if request_json["WA_TOD_RATE"]:
            WA_TOD_RATE = request_json['WA_TOD_RATE']

        if request_json["VALUE_TOD"]:
            VALUE_TOD = request_json['VALUE_TOD']

        if request_json["WA_TOM_RATE"]:
            WA_TOM_RATE = request_json['WA_TOM_RATE']

        if request_json["VALUE_TOM"]:
            VALUE_TOM = request_json['VALUE_TOM']

        if request_json["Rate"]:
            Rate = request_json['Rate']

        if request_json[" value_x"]:
            value_x = request_json[' value_x']

        if request_json[' value_y']:
            value_y = request_json[' value_y']

        if request_json['Brent']:
            Brent = request_json['Brent']

        if request_json['t-1']:
            t_1 = request_json['t-1']

        if request_json['t-2']:
            t_2 = request_json['t-2']

        logger.info(
            f'{dt} Data: curs_y={curs_y}, WA_TOD_RATE={WA_TOD_RATE}, VALUE_TOD={VALUE_TOD}, WA_TOM_RATE={WA_TOM_RATE}, VALUE_TOM={VALUE_TOM}, Rate={Rate}, value_x={value_x}, value_y={value_y}, Brent={Brent}, t_1={t_1}, t_2={t_2}')
        try:
            preds = model.predict(pd.DataFrame({"curs_y": [curs_y],
                                                "WA_TOD_RATE": [WA_TOD_RATE],
                                                "VALUE_TOD": VALUE_TOD,
                                                "WA_TOM_RATE": [WA_TOM_RATE],
                                                "VALUE_TOM": [VALUE_TOM],
                                                "Rate": [Rate],
                                                " value_x": [value_x],
                                                " value_y": [value_y],
                                                "Brent": [Brent],
                                                "t-1": [t_1],
                                                "t-2": [t_2]
                                                }, index=[date]))
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = preds[0]
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', debug=True, port=port)
