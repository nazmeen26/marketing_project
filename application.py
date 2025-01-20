import pickle
from flask import Flask, request, jsonify, render_template
from flask import Response
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

model=pickle.load(open("models/modelForPrediction.pkl", "rb"))
scaler=pickle.load(open("models/standardscalar.pkl", "rb"))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result=""

    if request.method=="POST":
        Age=float(request.form.get('Age'))
        Gender=float(request.form.get('Gender'))
        Income=float(request.form.get('Income'))
        AdSpend=float(request.form.get('AdSpend'))
        ClickThroughRate=float(request.form.get('ClickThroughRate'))
        ConversionRate=float(request.form.get('ConversionRate'))
        WebsiteVisits=float(request.form.get('WebsiteVisits'))
        PagesPerVisit=float(request.form.get('PagesPerVisit'))
        TimeOnSite=float(request.form.get('TimeOnSite'))
        SocialShares=float(request.form.get('SocialShares'))
        EmailOpens=float(request.form.get('EmailOpens'))
        EmailClicks=float(request.form.get('EmailClicks'))
        PreviousPurchases=float(request.form.get('PreviousPurchases'))
        LoyaltyPoints=float(request.form.get('LoyaltyPoints'))
        Campaign_Channel=float(request.form.get('Campaign_Channel'))
        Campaign_Type=float(request.form.get('Campaign_Type'))
        
        new_data=scaler.transform([[Age,Gender,Income,AdSpend,ClickThroughRate,ConversionRate,WebsiteVisits,PagesPerVisit,TimeOnSite,SocialShares,EmailOpens,EmailClicks,PreviousPurchases,LoyaltyPoints,Campaign_Channel,Campaign_Type]])
        predict=model.predict(new_data)
        
        if predict[0] ==1 :
            result = 'Converted'
        else:
            result ='Not-Converted'

        return render_template('single_prediction.html',result=result)


    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")




