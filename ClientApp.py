import os
import shutil

from flask import Flask,request,Response,jsonify,render_template,redirect
from flask import send_from_directory,send_file
from src.PredictionPipeline import PreditionClass
import zipfile



app = Flask(__name__)

PredictObj = PreditionClass()

@app.route("/Upload",methods=["POST"])
def InputPage():
    f = request.files['file']
    f.save("original.nii")

    result = PredictObj.Prediction()
    if result == True:

        # Delete the zip file if not needed
        os.remove("AllFiles.zip")
        filenames = os.listdir("Records/")
        shutil.rmtree("Records/")
        # return send_file(filenames[0],as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True,port=8000)