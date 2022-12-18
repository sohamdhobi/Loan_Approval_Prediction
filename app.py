
import os
from urllib import request
from flask import Flask,render_template,request
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.color_palette()

app=Flask(__name__)
#app.config["UPLOAD_PATH"]="C:\\Users\\soham dhobi\\Desktop\\flask practice"

@app.route("/")
def home():
   return render_template('upload.html')

global pa
app.config['UPLOAD_FOLDER']=os.path.join('static')
pa=app.config['UPLOAD_FOLDER']   

@app.route("/uploader",methods=["GET","POST"])
def uploader():
   if request.method == 'POST':
      global f
      
      f = request.files['file']
      f.save(os.path.join(pa,f.filename))
      #return render_template('dv.html')
      return dv() 

dataset=pd.DataFrame()
@app.route("/dv",methods=["POST","GET"])
def dv():
   if request.method == 'POST':
      global dataset
      dataset = pd.read_csv(f.filename)
      df=dataset.head()
      des=dataset.describe()
      nullseries=pd.Series(dataset.isnull().sum())
      dfnull=nullseries.to_frame(name='null values')
      
      # dfv=dataset.value_counts()
      # df_value=pd.DataFrame(dfv)
      # df_value_counts=df_value.reset_index()
      #df_value_counts.columns=["unique_vale","counts"]
      #picFolder = os.path.join()
      #app.config['UPLOAD_FOLDER']=os.path.join('static')
      dfi.export(df,os.path.join(pa,'head.png'))
      dfi.export(des,os.path.join(pa,'des.png'))
      dfi.export(dfnull,os.path.join(pa,'null_value.png'))
      #dfi.export(df_value_counts,os.path(pa, 'value_counts.png'))
      head = os.path.join(pa,'head.png')
      describe=os.path.join(pa,'des.png')
      null_values=os.path.join(pa,'null_value.png')
      # value_c=os.path.join(pa, 'value_counts.png')
      return render_template('dv.html' , head1=head, desc=describe,null_value=null_values)
      
   
# global df1
# df1=pd.DataFrame()
@app.route("/preprocessing",methods=["POST","GET"])
def remove_null():
   global dataset
   if request.method=='POST':
      dataset=dataset.dropna()
      dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
      dataset.replace(to_replace='3+',value=4, inplace=True)
      # dfh=dataset.head()
      # dfi.export(dfh,os.path.join(pa,'head2.png'))
      # head2=os.path.join(pa,'head2.png')
      s_p1=sns.pairplot(dataset,hue='Loan_Status',palette="bright")
      s_p1.savefig('static/fig1.png')
      #plot=os.getcwd()
      plot=os.path.join(pa,'fig1.png')

      #dfnull=nullseries.to_frame()
      #dfi.export(dfnull,os.path.join(pa,'rnull_value.png'))
      #rnull_values=os.path.join(pa,'rnull_value.png')
      return render_template('preprocess.html',fig=plot)
X=pd.DataFrame()
Y=pd.DataFrame()
@app.route("/model",methods=["POST","GET"])
def model_c():
   if request.method=='POST':
      global dataset
      dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
      dfh=dataset.head()
      dfi.export(dfh,os.path.join(pa,'head2.png'))
      head2=os.path.join(pa,'head2.png')
      global X,Y
      X=dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
      Y=dataset['Loan_Status']
      X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
      classifier=svm.SVC(kernel='linear')
      classifier.fit(X_train,Y_train)

      X_test_pridiction = classifier.predict(X_test)
      accuracy=accuracy_score(X_test_pridiction,Y_test)
      accuracy_per=accuracy*100
      pickle.dump(classifier,open('static/model.pkl','wb'))
      return render_template('model.html',cat=head2,acc=accuracy_per)
r=''

@app.route("/predict",methods=['POST','GET'])
def index():
   return render_template('predict.html')


@app.route("/result",methods=['POST','GET'])
def predict():
   #render_template('predict.html')
   if request.method=='POST':
      
      gender=request.form.get('gender',type=int)
      married=request.form.get('married',type=int)
      dependents=request.form.get('dependents',type=int)
      education=request.form.get('education',type=int)
      self_empolyed=request.form.get('self_empolyed',type=int)
      a_income=request.form.get('a_income')
      c_income=request.form.get('c_income')
      history=request.form.get('history',type=int)
      area=request.form.get('area',type=int)
      loan_amount=request.form.get('loan_amount')
      loan_term=request.form.get('loan_term')
      a_income=int(a_income)/10
      c_income=int(c_income)/10
      loan_amount=int(loan_amount)/1000
      loan_term=int(loan_term)*10
      global r
      model=pickle.load(open('static/model.pkl','rb'))
      result=model.predict(np.array([gender,married,dependents,education,self_empolyed,a_income,c_income,loan_amount,loan_term,history,area]).reshape(1,11))
      
      if result[0] == 1:
         result='Yes'
      else:
         result='NO'
      r=str(result)
   return render_template('predict.html',result=r)

# @app.route("/result",methods=["POST","GET"])
# def result():
#    if request.method=='POST':
#       global r

#       return render_template('result.html',res=r)



if __name__ == '__main__':
   app.run(debug=True)