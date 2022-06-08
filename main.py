from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import keras
import h5py
import cv2
import os
model=keras.models.load_model("covid19.h5")

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("covid.html")

@app.route("/predict",methods=["POST"])
def predict():
    x=request.files["myfile"]
    x.save(os.path.join("static/"+x.filename))
    img=cv2.imread("static/"+x.filename)
    img=cv2.resize(img,(100,100));
    img=img.reshape(1,100,100,3)
    result=model.predict(img)[0][0]

    if result==0:
        p="Patient Is NOT Suffered From COVID 19(-)"
    else:
        p="Patient Is Suffered From COVID19(+)"

    print(result)

    return render_template("covid.html",result=p,IMAGE_PATH="static/"+x.filename)



if __name__=="__main__":
    app.run(debug=True)