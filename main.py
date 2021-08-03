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
                if os.path.isdir("static/Records/Masked"):
                    shutil.rmtree("static/Records/Masked")
                if os.path.isdir("static/Records/Original"):
                    shutil.rmtree("static/Records/Original")
                    # os.rmdir("static/Records/")
            except:
                pass
            try:
                if not os.path.isdir("static/Records/Masked"):
                    os.makedirs("static/Records/Masked")
                if not os.path.isdir("static/Records/Original"):
                    os.makedirs("static/Records/Original")
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

            lis_original = sorted(os.listdir("static/Records/Original"))
            lis_masks = sorted(os.listdir("static/Records/Masked"))
            return render_template("results.html",imagesOriginal=lis_original,imagesMasked=lis_masks)

        else:
            return "Unsuccessfull prediction"
    except Exception as e:
        try:
            os.remove("ori1.nii")
        except:
            pass

        return "Exception : "+str(e)


if __name__ == "__main__":
    app.run(debug=True,port=8000)