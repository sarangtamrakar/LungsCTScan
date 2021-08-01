try:
    import shutil
    import os
    from flask import Flask, request, Response, jsonify, render_template, redirect
    from flask import send_from_directory, send_file
    from src.PredictionPipeline import PreditionClass
    import jinja2

except Exception as e:
    raise e




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

        try:
            try:
                if os.path.isdir("static/Records/"):
                    shutil.rmtree("static/Records/")
                    # os.rmdir("static/Records/")
            except:
                pass
            try:
                if not os.path.isdir("static/Records"):
                    os.makedirs("static/Records")
            except:
                pass

        except Exception as e:
            print(e)
            pass

        input_start = int(request.form['input_start'])
        input_end = int(request.form['input_end'])
        data = request.files['file']
        data.save("ori1.nii")
        result = PredictObj.Prediction(input_start,input_end)
        if result == True:
            try:
                os.remove("ori1.nii")
            except:
                pass

            lis = os.listdir("static/Records")
            return render_template("results.html",images=lis)

        else:
            return "Unsuccessfull prediction"
    except Exception as e:
        os.remove("ori1.nii")
        return "Exception : "+str(e)


if __name__ == "__main__":
    app.run(debug=True,port=8000)