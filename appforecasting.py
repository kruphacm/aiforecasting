import matplotlib.pyplot as plt
import mpld3
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask import Markup
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
app = Flask(__name__)
model1 = pickle.load(open('modelheartbeat.pkl', 'rb'))
model2 = pickle.load(open('modelbloodoxygen.pkl', 'rb'))
model3 = pickle.load(open('modelsystolic.pkl', 'rb'))
model4 = pickle.load(open('modeldiastolic.pkl', 'rb'))
model5 = pickle.load(open('modeltemperature.pkl', 'rb'))


@app.route('/')
def home():
    df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/forecasting%20data.csv")
    df['BLOOD PRESSURE']=df['BLOOD PRESSURE'].str.split("/")
    df['SYSTOLIC']=df['BLOOD PRESSURE'].str[0]
    df['DIASTOLIC']=df['BLOOD PRESSURE'].str[1]
    #combining all the  results
    orgdate=list(str(df['DATE'].tail(1)).strip().split(" "))
    date=orgdate[4][:orgdate[4].rfind("\n")]
    orgtime=list(str(df['TIME'].tail(1)).strip().split(" "))
    time=orgtime[4][:orgtime[4].rfind("\n")]
    #tempratue
    orgtemp=list(str(df['TEMPERATURE'].tail(1)).strip().split(" "))
    temperature=orgtemp[4][:orgtemp[4].rfind("\n")]
    p3=model5.predict([[temperature,temperature]])
    #systolic
    orgsys=list(str(df['SYSTOLIC'].tail(1)).strip().split(" "))
    systolic=orgsys[4][:orgsys[4].rfind("\n")]
    print(systolic)
    a=model3.predict([[systolic,systolic]])
    #diastolic
    orgdia=list(str(df['DIASTOLIC'].tail(1)).strip().split(" "))
    diastolic=orgdia[4][:orgdia[4].rfind("\n")] 
    b=model4.predict([[diastolic,diastolic]])
    #blood oxygen
    orgbo=list(str(df['BLOOD OXYGEN LEVEL'].tail(1)).strip().split(" "))
    bloodoxygen=orgbo[4][:orgbo[4].rfind("\n")]
    p1=model2.predict([[bloodoxygen,bloodoxygen]])
    #heart beat
    orghb=list(str(df['HEART BEAT'].tail(1)).strip().split(" "))
    heartbeat=orghb[4][:orghb[4].rfind("\n")]
    p=model1.predict([[heartbeat,heartbeat]])
    #condition for printing 
    print(p,a,b,p1,p3)
    output ="<br><br><br><p style='padding left:5%;'><center>REPORT</center></p><br><br>Date: "+str(date)+"<br><br>Time:"+str(time)+"<br><br><br>Readings<br><br>HEARTBEAT: "+str(heartbeat)+"<br><br>BLOOD PRESSURE(SYSTOLIC): "+str(systolic)+"<br><br>BLOOD PRESSURE(DIASTOLIC): "+str(diastolic)+"<br><br>BLOOD OXYGEN: "+str(bloodoxygen)+"<br><br>TEMPRATURE: "+str(temperature)+"<br><br>NORMAL RANGE<br><br>HEARTBEAT: 60 to 100 bpm<br><br>BLOOD PRESSURE(SYSTOLIC): 90 to 120<br><br>BLOOD PRESSURE(DIASTOLIC): 60 to 80<br><br>BLOOD OXYGEN: 95.0% to 99.9%<br><br>TEMPRATURE: 97.7 F to 99.0 F<br><br>RESULT<br>"
    fHB,fS,fD,fBO,fT=(float(heartbeat)/100)*10,(float(systolic)/120)*10,(float(diastolic)/80)*5,(float(bloodoxygen)/100)*5,(float(temperature)/97)*5
    if p3==0.0 and a==0.0 and b==0.0 and p1==0.0 and p==0.0:
        
        output+="<p style='color:green;background-color:white;'>NORMAL</p><br>MEDICATION AND DIET<br><br>FOOD SUGGESTION FOR NORMAL RANGE<br><br>Normal HeartBeat:Banana, melons, orange ,sweet potatoes, dairy, whole grains, chicken(8-ounce glass of water)<br><br>Normal BP:egg, chicken, nuts and seeds, fruits and vegetables.<br><br>Normal Temprature:Hot water, water rich foods like cucumber and water melon, green leafy vegetables like spinach, kale, broccoli.<br><br>Normal Blood Oxygen:Beetroot, garlic, leafy greens, pomegranate, cruciferous vegetables, sprouts, meat, nuts, seeds, dates, carrots, banana.<br><br>Medication: Follow Your Regular Medication.<br><br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(heartbeat)+" to "+str(int(int(heartbeat)+fHB))+"<br>predicted systolic: "+str(systolic)+" to "+str(int(float(systolic)+fS))+"<br>predicted diastolic: "+str(diastolic)+" to "+str(int(float(diastolic)+fD))+"<br>predicted Blood Oxygen: "+str(bloodoxygen)+" to "+str(round((float(bloodoxygen)+fBO),2))+"<br>predicted Temprature: "+str(temperature)+" to "+str(round((float(temperature)+fT),2))+"<p style='font-size:100%;color:red;'>These predictions will be apllicable when the above diet followed.</p>"
    if p3==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL HEARTBEAT</p><br>MEDICATION AND DIET<br><br>FOOD SUGGESTION FOR ABNORMAL RANGE"
        if heartbeat>100:
            output+="<br><br>Above Normal Range:Omega-3 fatty acids, found in fish, lean meats, nuts, grains and legume. Phenols and tannins found in tea, coffee. and red wine(in moderation).Vitamin A, found in greens. Whole grains. Vitamin C in bean sprouts.<br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(int(int(heartbeat)-fHB))+" to "+str(heartbeat)+"<br><br>Disease Related to High Pulse<br><br>Tachycardia:<br><br>disease: Stroke, Heart failure, Sudden death, Blood clots<br><br>Symptoms:<br>a fast pulse,chest pain,confusion,dizziness,low blood pressure,lightheadedness,heart palpitations,shortness of breath,sudden weakness,fainting,a loss of consciousness and cardiac arrest, in some cases<br><br>Prevention<br>Eating a heart-healthy diet,Staying physically active and keeping a healthy weight,Avoiding smoking,Limiting or avoiding caffeine and alcohol,Reducing stress, as intense stress and anger can cause heart rhythm problems and Using over-the-counter medications with caution, as some cold and cough medications contain stimulants that may trigger a rapid heartbeat<br><br>Treatment<br>treatment options for tachycardia will depend on various factors like the cause,the age of the person,their overall health like Vagal maneuvers,Medication:amiodarone (Cordarone), sotalol (Betapace), and mexiletine (Mexitil),calcium channel blockers, such as diltiazem (Cardizem) or verapamil (Calan),beta-blockers, such as propranolol (Inderal) or metoprolol (Lopressor)and blood thinners, such as warfarin (Coumadin) or apixaban (Eliquis),Cardioversion and defibrillators(electric shockTrusted Source)Radiofrequency catheter ablation and Surgery"
        if heartbeat<60:
            output+="<br><br>Below Normal Range:Chia seeds, flax seeds, and hemp seeds, greens vegetables, whole grains, berries, avocados, fatty fish and fish oil, walnuts, beans, dark chocolate, tomatoes, almonds, garlic, olive oil.<br><br>FORECASTED PREDICTION:<br>predicted heart beat: "+str(heartbeat)+" to "+str(heartbeat+fHB)+"<br><br>Disease Related to low Pulse<br><br>Bradycardia<br><br>Symptoms:<br>Fatigue or feeling weak,Dizziness or lightheadedness,Confusion,Fainting (or near-fainting) spells,Shortness of breath,Difficulty when exercising and Cardiac arrest (in extreme cases)<br><br>Diseases:<br>In some cases, slow heartbeat may be a symptom of a serious or life-threatening condition that should be immediately evaluated in an emergency setting. These conditions include:Cardiogenic shock (shock caused by heart damage and ineffective heart function),Congestive heart failure (deterioration of the heart’s ability to pump blood),Dissecting aortic aneurysm (life-threatening bulging and weakening of the aortic artery wall that can burst and cause severe hemorrhage),Myocardial infarction (heart attack),Myocarditis (infection of the middle layer of the heart wall),Overdose, including cumulative overdose, of certain cardiac medications,Pericarditis (infection of the lining that surrounds the heart) and Trauma.<br><br>Treatment:<br>Borderline or occasional bradycardia may not require treatment.,Severe or prolonged bradycardia can be treated in a few ways. For instance, if medication side effects are causing the slow heart rate, then the medication regimen can be adjusted or discontinued andIn many cases, a pacemaker can regulate the heart’s rhythm, speeding up the heart rate as needed."
    if a==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD PRESSURE(SYSTOLIC)</p>"
        if systolic>120:
            output+="<br><br>Above Normal Range:Plenty of fruit, vegetables and wholegrains. Fish and seafood, legumes such as beans and lentils , nuts and seeds, unflavored milk, yoghurt and cheese, herbs and spices.<br><br>FORECASTED PREDICTION:<br>Predicted Systolic: "+str(int(int(systolic)-fS))+" to "+str(systolic)+"<br><br> Disease Realted to High BP<br><br><ul><li>hypertension</li><li>Heart attack<br>Symptoms:<br>Nausea, indigestion, heartburn or abdominal pain,Nausea, indigestion, heartburn or abdominal pain,Shortness of breath,Cold sweat,Fatigue and Lightheadedness or sudden dizziness<br><br>Tips<br>Pain or pressure in the chest is the most common symptom of a heart attack. However, pain or discomfort in the arms, back, neck or jaw can also be a sign — and so can shortness of breath, nausea or light-headedness. If you experience one or more of these warning signs, CALL 911 immediately, even if you’re not sure it’s a heart attack.<br><br>Medications:<br>Medications:Aspirin,Thrombolytics,Antiplatelet agents,Heparin,Pain relievers,Nitroglycerin,Beta blockers,ACE inhibitors,Statins and Surgical & other procedures like Coronary angioplasty & stenting & Coronary artery bypass surgery</li><li>Stroke,Ischemic stroke,Transient ischemic attack (TIA) and Hemorrhagic stroke<br><br>Symptoms:<br>Weakness or Numbness of the face, arm, or leg,Confusion or trouble speaking or understanding others,Difficulty in vision,Difficulty in walking or loss of balance or coordination and Severe headache with unknown cause.<br><br>Tips<br>,Stop smoking,Discuss weight monitoring with your doctor,Check your legs, ankles and feet for swelling daily,Eat a healthy diet,Restrict sodium in your diet,Maintain a healthy weight,Consider getting vaccinations,Limit saturated or 'trans' fats in your diet, alcohol and fluids,Be active,Reduce stress and Sleep easy<br>Medications:Angiotensin-converting enzyme (ACE) inhibitors- enalapril (Vasotec), lisinopril (Zestril) and captopril (Capoten),Angiotensin II receptor blockers- losartan (Cozaar) and valsartan (Diovan),Beta blockers- carvedilol (Coreg), metoprolol (Lopressor) and bisoprolol (Zebeta),Diuretics-   furosemide (Lasix),Aldosterone antagonists- spironolactone (Aldactone) and eplerenone (Inspra),Inotropes and Digoxin (Lanoxin)<br><br>Surgery and medical devices:<br>Coronary bypass surgery,Heart valve repair or replacement,Implantable cardioverter-defibrillators (ICDs),Cardiac resynchronization therapy (CRT), or biventricular pacing.,Ventricular assist devices (VADs)and Heart transplant.</li></ul>"
        if systolic<90:
            output+="<br><br>Below Normal Range: Eat salty food like cottage cheese, canned soup, tuna, olives, drink caffeine, egg, chicken, fish like salmon, asparagus, broccoli, liver and legumes such as lentils and chickpeas.<br><br>FORECASTED PREDICTION:<br>Predicted Systolic: "+str(systolic)+" to "+str(int(int(systolic)+fS))+"<br><br> Disease related to hypo tension<br><ul><li>Hypotensi<br><br>Symptoms<br>Dizziness or lightheadedness,Fainting,Blurred or fading vision,Nausea,Fatigue and Lack of concentration<br><br>Tips<br>If you have signs or symptoms of shock, seek emergency medical help,If you have consistently low blood pressure readings but feel fine, your doctor will likely just monitor you during routine exams,Even occasional dizziness or lightheadedness may be a relatively minor problem,Drink more water, less alcohol,Pay attention to your body positions,Eat small, low-carb meals and Exercise regularly<br><br>Medication<br>Use more salt,Drink more water,Wear compression stockings and Medications- fludrocortisones, midodrine (Orvaten)</li><li>Other diseases can be:<br>Low blood pressure on standing up (orthostatic or postural) hypotension) - This is a sudden drop in blood pressure when you stand up from a sitting position or after lying down.<br><br>Low blood pressure after eating (postprandial hypotension) - This drop in blood pressure occurs one to two hours after eating and affects mostly older adults.<br><br>Low blood pressure from faulty brain signals (neurally mediated hypotension) - This disorder, which causes a blood pressure drop after standing for long periods, mostly affects young adults and children. It seems to occur because of a miscommunication between the heart and the brain.<br><br>Low blood pressure due to nervous system damage (multiple system atrophy with orthostatic hypotension) - Also called Shy-Drager syndrome, this rare disorder has many Parkinson disease-like symptoms. It causes progressive damage to the autonomic nervous system, which controls involuntary functions such as blood pressure, heart rate, breathing and digestion. It's associated with having very high blood pressure while lying down.</li></ul>"
    if b==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD PRESSURE(DIASTOLIC</p>"
        if diastolic>80:
            output+="<br><br>Above Normal Range:Plenty of fruit, vegetables and wholegrains. Fish and seafood, legumes such as beans and lentils , nuts and seeds, unflavored milk, yoghurt and cheese, herbs and spices.<br><br>FORECASTED PREDICTION:<br>Predicted Diastolic"+str(int(int(diastolic)-fD))+" to "+str(diastolic)+"<br><br> Disease Realted to High BP<br><br><ul><li>hypertension</li><li>Heart attack<br>Symptoms:<br>Nausea, indigestion, heartburn or abdominal pain,Nausea, indigestion, heartburn or abdominal pain,Shortness of breath,Cold sweat,Fatigue and Lightheadedness or sudden dizziness<br><br>Tips<br>Pain or pressure in the chest is the most common symptom of a heart attack. However, pain or discomfort in the arms, back, neck or jaw can also be a sign — and so can shortness of breath, nausea or light-headedness. If you experience one or more of these warning signs, CALL 911 immediately, even if you’re not sure it’s a heart attack.<br><br>Medications:<br>Medications:Aspirin,Thrombolytics,Antiplatelet agents,Heparin,Pain relievers,Nitroglycerin,Beta blockers,ACE inhibitors,Statins and Surgical & other procedures like Coronary angioplasty & stenting & Coronary artery bypass surgery</li><li>Stroke,Ischemic stroke,Transient ischemic attack (TIA) and Hemorrhagic stroke<br><br>Symptoms:<br>Weakness or Numbness of the face, arm, or leg,Confusion or trouble speaking or understanding others,Difficulty in vision,Difficulty in walking or loss of balance or coordination and Severe headache with unknown cause.<br><br>Tips<br>,Stop smoking,Discuss weight monitoring with your doctor,Check your legs, ankles and feet for swelling daily,Eat a healthy diet,Restrict sodium in your diet,Maintain a healthy weight,Consider getting vaccinations,Limit saturated or 'trans' fats in your diet, alcohol and fluids,Be active,Reduce stress and Sleep easy<br>Medications:Angiotensin-converting enzyme (ACE) inhibitors- enalapril (Vasotec), lisinopril (Zestril) and captopril (Capoten),Angiotensin II receptor blockers- losartan (Cozaar) and valsartan (Diovan),Beta blockers- carvedilol (Coreg), metoprolol (Lopressor) and bisoprolol (Zebeta),Diuretics-   furosemide (Lasix),Aldosterone antagonists- spironolactone (Aldactone) and eplerenone (Inspra),Inotropes and Digoxin (Lanoxin)<br><br>Surgery and medical devices:<br>Coronary bypass surgery,Heart valve repair or replacement,Implantable cardioverter-defibrillators (ICDs),Cardiac resynchronization therapy (CRT), or biventricular pacing.,Ventricular assist devices (VADs)and Heart transplant.</li></ul>"
        if diastolic<60:
            output+="<br><br>Below Normal Range: Eat salty food like cottage cheese, canned soup, tuna, olives, drink caffeine, egg, chicken, fish like salmon, asparagus, broccoli, liver and legumes such as lentils and chickpeas.<br><br>FORECASTED PREDICTION:<br>Predicted Diastolic"+str(diastolic)+" to "+str(int(int(diastolic)+fD))+"<br><br> Disease related to hypo tension<br><ul><li>Hypotensi<br><br>Symptoms<br>Dizziness or lightheadedness,Fainting,Blurred or fading vision,Nausea,Fatigue and Lack of concentration<br><br>Tips<br>If you have signs or symptoms of shock, seek emergency medical help,If you have consistently low blood pressure readings but feel fine, your doctor will likely just monitor you during routine exams,Even occasional dizziness or lightheadedness may be a relatively minor problem,Drink more water, less alcohol,Pay attention to your body positions,Eat small, low-carb meals and Exercise regularly<br><br>Medication<br>Use more salt,Drink more water,Wear compression stockings and Medications- fludrocortisones, midodrine (Orvaten)</li><li>Other diseases can be:<br>Low blood pressure on standing up (orthostatic or postural) hypotension) - This is a sudden drop in blood pressure when you stand up from a sitting position or after lying down.<br><br>Low blood pressure after eating (postprandial hypotension) - This drop in blood pressure occurs one to two hours after eating and affects mostly older adults.<br><br>Low blood pressure from faulty brain signals (neurally mediated hypotension) - This disorder, which causes a blood pressure drop after standing for long periods, mostly affects young adults and children. It seems to occur because of a miscommunication between the heart and the brain.<br><br>Low blood pressure due to nervous system damage (multiple system atrophy with orthostatic hypotension) - Also called Shy-Drager syndrome, this rare disorder has many Parkinson disease-like symptoms. It causes progressive damage to the autonomic nervous system, which controls involuntary functions such as blood pressure, heart rate, breathing and digestion. It's associated with having very high blood pressure while lying down.</li></ul>"
    if p1==1.0:
        output+="<p style='color:red;background-color:white;'>ABNORMAL BLOOD OXYGEN</p>"
        if bloodoxygen<90:
            output+="<br><br>Below Normal Range:Cayenna pepper, beets, berries, fatty fish, pomegranates, garlic, walnuts, grapes, turmeric, spinach, citrus fruit , chocolate, ginger.<br><br>FORECASTED PREDICTION:<br>Predicted Blood Oxygen level: "+str(bloodoxygen)+" to "+str(round((float(bloodoxygen)+fBO),2))+"<br><br>Disease Related to low N=bLood Oxygen<br><br>Symptoms:<br>apid breathing,shortness of breath,fast heart rate,coughing or wheezing,sweating,confusion and changes in the color of your skin<br><br>Treatment:<br>Medication-inhaler,oxygen gas,liquid oxygen,oxygen concentrators and hyperbaric oxygen therapy<br><br>Tips/Prevention:<br>Stop smoking, and avoid secondhand smoke or environmental irritants,Eat foods rich in antioxidants,Get vaccinations like the flu vaccine and the pneumonia vaccine. This can help prevent lung infections and promote lung health,Exercise more frequently, which can help your lungs function properly and Improve indoor air quality. Use tools like indoor air filters and reduce pollutants like artificial fragrances, mold, and dust."
        if bloodoygen >100:
            output+="<br><br>FORECASTED PREDICTION:<br>Predicted Blood Oxygen level: "+str(round((float(bloodoxygen)-fBO),2))+" to "+str(bloodoxygen)+"Disease Related to High Blood Oxygen<br><br>Oxygen Toxicity<br><br>Symptoms<br>Coughing,Mild throat irritation,Chest pain,Trouble breathing,Muscle twitching in face and hands,Dizziness,Blurred vision,Nausea,A feeling of unease,Confusion and Convulsions (seizure)<br><br>Treatment<br>Your lungs may take weeks or more to recover fully on their own. If you have a collapsed lung, you may need to use a ventilator for a while. Your healthcare provider will tell you more about any other kinds of treatment."

    if p==1.0:
        output="<p style='color:red;background-color:white;'>ABNORMAL TEMPERATURE</p>"
        if temprature>99:
            output+="<br><br>Above Normal Range: Chicken soup, garlic, coconut water, hot tea, honey, ginger, spicy foods, bananas, oatmeal, yogurt, fruits like strawberries, cranberries, blueberries, blackberries, avocados, greeny vegetables, salmon.<br><br>FORECASTED PREDICTION:<br>Predicted Temperature: "+str(temperature)+" to "+str(round((float(temperature)+fT),2))+"<br><br>Hyperpyrexia<br><br>Symptoms<br>increased thirst,extreme sweating,dizziness,muscle cramps,fatigue and weakness,nausea and light-headedness<br><br>Treatement:<br>a cool bath or cold, wet sponges put on the skin,liquid hydration through IV or from drinking and fever-reducing medications, such as dantrolene"
        if temperature<97.7:
            output+="<br><br>Below Normal Range:Hot tea or coffee, soup, roasted veggies, protein and fats like nuts, avocados, seeds ,olives, salmon, hard-boiled eggs, iron like shellfish, red meat, beans, broccoli.<br><br>FORECASTED PREDICTION:<br>Predicted Temperature: "+str(round((float(temperature)-fT),2))+" to "+str(temperature)+"<br><br>Disease Related to Low temperature<br><br>Hypothermia :<br><br>Symptoms:<br>shivering,slow, shallow breath,slurred or mumbled speech,a weak pulse,poor coordination or clumsiness,low energy or sleepiness,confusion or memory loss and loss of consciousness<br><br>complications:<br>frostbite, or tissue death, which is the most common complication that occurs when body tissues freeze,chilblains, or nerve and blood vessel damage,gangrene, or tissue destruction,trench foot, which is nerve and blood vessel destruction from water immersion and Hypothermia can also cause death.<br><br>Medications:<br> antidepressants, sedatives, and antipsychotic ,warm fluids, often saline, injected into the veins, Airway rewarming.<br><br>Tips/prevention:<br>Handle the person with care,Remove the person’s wet clothing,Apply warm compresses and Monitor the person’s breathing."

    output=Markup(output)

    return render_template('AI FORECASTING.html', prediction_text=output) 
 
@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/forecasting%20data.csv")
    df['BLOOD PRESSURE']=df['BLOOD PRESSURE'].str.split("/")
    df['SYSTOLIC']=df['BLOOD PRESSURE'].str[0]
    df['DIASTOLIC']=df['BLOOD PRESSURE'].str[1]
    int_features = [x for x in request.form.values()]
    index1=df.DATE[df.DATE == int_features[0]].index.tolist()
    index=df.DATE[df.DATE == int_features[1]].index.tolist()
    print(int_features,index,index1)
    I1,I2=index1[0],index[len(index)-1]
    if(int_features[2]=='HEART BEAT'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['HEART BEAT'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.xlabel("Date")
        plt.ylabel("Heart beat")
        plt.legend(['Heart Beat'])
    elif(int_features[2]=='BLOOD OXYGEN'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['BLOOD OXYGEN LEVEL'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.xlabel("Date")
        plt.ylabel("Blood Oxygen")
        plt.legend(['Blood Oxygen'])
    elif(int_features[2]=='BLOOD PRESSURE'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['SYSTOLIC'][I1:I2])
        xpoints1 = np.array(df['DATE'][I1:I2])
        ypoints1 = np.array(df['DIASTOLIC'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.plot(xpoints1, ypoints1)
        plt.legend(['systolic','diastolic'])
        plt.xlabel("Date")
        plt.ylabel("Blood Pressure")
    elif (int_features[2]=='TEMPERATURE'):
        xpoints = np.array(df['DATE'][I1:I2])
        ypoints = np.array(df['TEMPERATURE'][I1:I2])
        figure= plt.figure()
        figure.set_figwidth(10)
        figure.set_figheight(10)
        plt.plot(xpoints, ypoints)
        plt.xlabel("Date")
        plt.ylabel("temperature")
        plt.legend(['Temperature'])
    elif(int_features[2]=='ALL'):
        figure, axis = plt.subplots(2,2)
        figure.set_figwidth(10)
        figure.set_figheight(10)
        I1,I2=index1[0],index[len(index)-1]
        xpoints = np.array(df['DATE'][I1:I2])       
        ypoints = np.array(df['HEART BEAT'][I1:I2])        
        xpoints1 = np.array(df['DATE'][I1:I2])
        ypoints1= np.array(df['BLOOD OXYGEN LEVEL'][I1:I2])
        xpoints2 = np.array(df['DATE'][I1:I2])
        ypoints2 = np.array(df['SYSTOLIC'][I1:I2])       
        xpoints3 = np.array(df['DATE'][I1:I2])
        ypoints3 = np.array(df['DIASTOLIC'][I1:I2])       
        xpoints4 = np.array(df['DATE'][I1:I2])
        ypoints4 = np.array(df['TEMPERATURE'][I1:I2])
        axis[0,0].plot(xpoints, ypoints)        
        axis[0,0].legend(['Heart Beat'])        
        axis[0,1].plot(xpoints1, ypoints1)                
        axis[0,1].legend(['Blood Oxygen'])        
        axis[1,0].plot(xpoints2, ypoints2)        
        axis[1,0].plot(xpoints3, ypoints3)        
        axis[1,0].legend(['systolic','diastolic'])      
        axis[1,1].plot(xpoints4, ypoints4)
        axis[1,1].legend(['Temperature'])  
    plt_html = mpld3.fig_to_html(figure)

    return '''<!DOCTYPE html>
<html >
<head>
   <meta charset="utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <script src="html2pdf.bundle.min.js"></script>

  <title>AI Predictions</title>
  
</head>
    <style>
    body::-webkit-scrollbar
    {
        display:none;
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;
        
    }
        body
        {
            background-color:rgba(	255, 159, 88);
            filter: brightness(85%)
        }
.navbar,.navbar1,.desc{
  overflow: hidden;

  font-family: sans-serif,monospace;
}

.navbar a {
  float: left;
  font-size: 16px;
  color: white;
  text-align: center;
   padding: 14px 16px;
  text-decoration: none;
}
img
    {
        padding-left: 4%;
        padding-top: 2%;
        padding-bottom: 2%;
    }
.cropped1
    {
           padding-left: 2%;
        vertical-align: middle;
        font-size: 150%;
    }
.navbar{
    background-color: #555;

        color: white;
        font-size: auto;
        font-size: 80%;
    font-style: italic;
    font-family:cursive;
        padding-left: 2%;
 
    }
div.scrollmenu {
    padding-top: 1%;
  background-color: #333;
  overflow: auto;
  white-space: nowrap;
}

div.scrollmenu a {
  display: inline-block;
  color: white;
  text-align: center;
  padding: 14px;
  text-decoration: none;
    font-family: cursive,monospace;
}

div.scrollmenu a:hover {
  background-color: #777;
}
.scrollmenu {
  background-color: #eee;
  width: 100%;
  height: 100%;
  border: 1px dotted black;
  overflow-y: scroll; /* Add the ability to scroll */
}

/* Hide scrollbar for Chrome, Safari and Opera */
.scrollmenu::-webkit-scrollbar {
    width: 2%;
}

/* Hide scrollbar for IE, Edge and Firefox */
.scrollmenu {
  -ms-overflow-style: none;  /* IE and Edge */
  scrollbar-width: 2%;  /* Firefox */
}
        .mySlides
        {
            display: none;
        }
    </style>
  
<script>
var slideIndex = 0;
showDivs(slideIndex);

function plusDivs(n) {
  showDivs(slideIndex += n);
}

function showDivs(n) {
  var i;
    
  var x = document.getElementsByClassName("mySlides");
  var y = document.getElementsByClassName("mySlides1");
  if (n > x.length) {slideIndex = 1}
  if (n < 1) {slideIndex = x.length}
  for (i = 0; i < x.length; i++) {
    x[i].style.display = "none";  
  }
  for (i = 0; i < y.length; i++) {
    y[i].style.display = "none";  
  }    
  x[slideIndex-1].style.display = "block";  
}
</script>
     <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.9.2/html2pdf.bundle.js"></script>
    <script>
        
        window.onload = function () {
    document.getElementById("create_pdf")
        .addEventListener("click", () => {
            const invoice = this.document.getElementById("report");
            console.log(invoice);
            console.log(window);
            var opt = {
                margin: 1,
                filename: 'myreport.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2 },
                jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
            };
            html2pdf().from(invoice).set(opt).save();
        })
}
    </script>
<body>
<div class='navbar' id='title'>
    <div></div>
    <div></div>
    <p style="font-size: 140%"><img class="cropped1" src="https://drive.google.com/thumbnail?id=1JQ6epr36ugrVF7cuTlYMw9kL7J-pZrfd" width=5%; height="5%;">&nbsp HEALTH COMPANION</p></div>
    <div class="scrollmenu">
             <a href="#home" >HOME</a>
             <a href="#news" >BLOG</a>
            <a href='Consulting%20doctors.html' >CONSULTING</a>
             <a href='' >CHECK MY HEALTH</a>
             <a href="AI%20predictions.html" >AI PREDICTIONS</a>
             <a href="Hospital.html">HOSPITAL CONSULTING</a>      
             <a href="" >SCAN CENTRE AND HOSPITALS</a>
                 <a href="HealthyTips.html">HEALTH TIPS</a>
                 <a href="food1-1.html">FOOD INFO</a>
                  <a href="Disease%20information.html">DISEASE INFO</a>
             </div>
  <br>
  <br>
  <br>

    <div class="w3-content w3-section" style='background-color: #333; color: white;' >
        <br><br>
       <p style="text-align: center;">CLICK RIGHT ARROW TO SEE THE STEPS</p>
         <!-- Navigation arrows -->  
        <a class="left" onclick="plusDivs(-1)" style="padding-left: 5%;  padding-top: -15; font-size: 150%;"><b>❮</b></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        <a class="right" onclick="plusDivs(1)" style="padding-right: 5%; float: right; padding-top: -15; font-size: 150%;">❯</a>
        <div><img class="mySlides1" src="https://www.log-hub.com/wp-content/uploads/2017/12/forecating_process.png" width='90%;' ></div>
         
       <div class='id1'><img class="mySlides" src="https://raw.githubusercontent.com/kruphacm/mini-project/main/FORECAST1.png"  width='90%'></div>
       <div class='id2'><img class="mySlides"  src="https://raw.githubusercontent.com/kruphacm/mini-project/main/FORECAST2.png" width='90%'></div>
      <div class='id3'> <img class="mySlides"  src="https://raw.githubusercontent.com/kruphacm/mini-project/main/FORECAST3.png"width='90%'></div>     
   
   
</div><br><br>
    <div class='form' style='color: black; background-color: #333;padding: 10%; '>
         <br><br>

        <div style='padding-left:5%; padding-right: 20%; font-size: 150%;background-color: white;' id="report">
     {{ prediction_text }}
         <br>
         <br></div>
     </div>
    <div  style="background-color: #333;padding-left: 45%">
    <br><br>
        <input type="button" id="create_pdf" value="GENERATE PDF" > <br><br><br>  <br><br></div> 
    <br><br><br>
    <div style="background-color: #333;padding: 10%; color:white;">
      <form action="{{ url_for('predict')}}"method="post" style='color: white;  text-align:center; font-size: 150%;'>
          <p >the Date Startes from 01-01-2021 to 30-04-2021</p>
          <p style="color: red;">enter the starting date and ending date within 20 days limit.</p><br><br>
        <p>Enter Starting Date:<input type="text" name="Enter the Starting Date" placeholder="Enter the Starting Date" required="required" /></p><br><br>
        <p>Enter the Ending Date:<input type="text" name="Enter the Ending Date" placeholder="Enter the Ending Date" required="required" /></p><br><br>
          <label for="cars">Choose a parameter:</label>
          <select id="cars" name="cars">
  <option value="HEART BEAT">HEART BEAT</option>
  <option value="BLOOD OXYGEN">BLOOD OXYGEN</option>
  <option value="BLOOD PRESSURE">BLOOD PRESSURE</option>
  <option value="TEMPERATURE">TEMPERATURE</option>
  <option value="ALL">ALL THE ABOVE</option>
</select><br><br><br><br><br>
        <button type="submit" class="btn btn-primary btn-block btn-large" style='background-color:lightsalmon; color: white; font-size: 100%; padding: 1%;'>&nbsp&nbsp &nbsp SHOW GRAPH&nbsp&nbsp &nbsp  </button><br><br>
    </form>
    </div>
    <div style='color: white; background-color: #333;text-align: center; font-size: 150%;'>
         <br><br>
     <p>GRAPH</p>
 <div style="background-color: #333;padding-left: 10%;padding-right: 10%; ">
        <div><div style="padding-left: 7%; background: white;">''' + plt_html + '''</div></div><br><br>
     <div style="background-color: white;">
     <p style="color:red; ">NOTE: this can be used only once so click the button below  to view the graph again</p></div></div><br><br>
        <a href="https://aiforecasting.herokuapp.com/" style="color:white; background-color:lightsalmon;padding: 2%;text-decoration: none;">CLICK HERE</a><br><br><br>
         <br>
         <br>
     </div>
    </body>
</html>'''
    


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
