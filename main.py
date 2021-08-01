import os
import shutil

from flask import Flask,request,Response,jsonify,render_template,redirect
from flask import send_from_directory,send_file
from src.PredictionPipeline import PreditionClass
import zipfile



app = Flask(__name__)

PredictObj = PreditionClass()


@app.route("/",methods=['GET'])
def HomePage():
    return render_template("index.html")


@app.route("/UploadInfo",methods=["POST"])
def InputPage():
    PatientID = request.form['PatientID']
    Email = request.form['Email']
    print(PatientID)
    print(Email)
    return render_template("input.html")

@app.route("/UploadImage",methods=["POST"])
def ImagesPrediction():
    try:
        try:
            os.remove('ori1.nii')
        except:
            pass

        input_start = int(request.form['input_start'])
        input_end = int(request.form['input_end'])
        data = request.files['file']
        data.save("ori1.nii")
        result = PredictObj.Prediction(input_start,input_end)
        if result == True:
            os.remove("ori1.nii")
            return render_template("results.html")
        else:
            return "Unsuccessfull prediction"
    except Exception as e:
        os.remove("ori1.nii")
        return "Exception : "+str(e)


if __name__ == "__main__":
    app.run(debug=True,port=8000)