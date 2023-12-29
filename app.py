import os
from flask import Flask, flash, request, redirect, url_for
from flask import render_template
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import statistics as stat

from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDRegressor, LinearRegression
from regressor import *
from classifier import *
from sklearn.model_selection import train_test_split
import time as tm
import statsmodels.api as sm

#Import des librairies Scikit-learn pour la regression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,median_absolute_error,confusion_matrix
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel,DotProduct
from sklearn.model_selection import learning_curve,cross_val_score
import time as tm

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import  GaussianProcessRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor


#Import des librairies Scikit-learn pour la regression

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score,confusion_matrix,classification_report
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel,DotProduct

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import  GaussianProcessClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


###############################################################################
#################### CONNEXION AU BASE DE DONNEE ##############################
import sqlite3
#creation de base de donnee
conn = sqlite3.connect('prevision.db')
c = conn.cursor()
#Creation des table 

c.execute('''CREATE TABLE IF NOT EXISTS login (username text, password text)''')

c.execute('''CREATE TABLE IF NOT EXISTS historique
             (Nom text, Date_Creation date, Type_data text, Type_traitement text, Score Real, Modele INTEGER, Target text,Prediction real)''')
#Insertion login

#c.execute("INSERT INTO login VALUES ('ADMIN','root')")

# Save (commit) the changes
conn.commit()
#Representation des modeles
####  Regression  ####

def regression():

    global Ridge,SGD,SVM,KNN,Tree,Extratree,RNA,RForest,GBoosting,AdaBoost,Gaussian

    kernel_gp = DotProduct() + WhiteKernel()

    preprocessor=make_pipeline(PolynomialFeatures(2, include_bias=False),StandardScaler(), SelectKBest(f_classif,k='all'))

    Ridge=make_pipeline(preprocessor, Ridge(random_state=0))
    SGD=make_pipeline(preprocessor,SGDRegressor(loss='squared_loss',random_state=0,max_iter=1000, tol=1e-3))
    SVM=make_pipeline(preprocessor,SVR(kernel='linear'))

    KNN=make_pipeline(preprocessor, KNeighborsRegressor(n_neighbors=2,weights='uniform',leaf_size=30,algorithm='auto'))
    Tree=make_pipeline(preprocessor,DecisionTreeRegressor(max_features=5,max_depth=4, random_state=0,min_samples_split=5))#,min_samples_split=5
    Extratree=make_pipeline(preprocessor,ExtraTreesRegressor(max_features=5,random_state=0,max_depth=4))

    RNA=make_pipeline(preprocessor, MLPRegressor(solver='lbfgs',hidden_layer_sizes=(8,),random_state=1,max_iter=1000))
    RForest=make_pipeline(preprocessor,RandomForestRegressor(random_state=0))
    GBoosting=make_pipeline(preprocessor,GradientBoostingRegressor(random_state=0))
    AdaBoost=make_pipeline(preprocessor, AdaBoostRegressor(random_state=0,n_estimators=40, loss='linear',learning_rate=1.0))
    Gaussian=make_pipeline(preprocessor,GaussianProcessRegressor(kernel=kernel_gp,n_restarts_optimizer=0, normalize_y=True, alpha=.5))#1e-20

##### CLassification  ####
def classification():

    global Ridge,SGD,SVM,KNN,Tree,Extratree,RNA,RForest,GBoosting,AdaBoost,Gaussian

    kernel_gp = DotProduct() + WhiteKernel()

    preprocessor=make_pipeline(PolynomialFeatures(2, include_bias=False),StandardScaler(), SelectKBest(f_classif,k=6))

    Ridge=make_pipeline(preprocessor, RidgeClassifier(random_state=0))
    SGD=make_pipeline(preprocessor,SGDClassifier(loss='squared_loss',random_state=0,max_iter=1000, tol=1e-3))
    SVM=make_pipeline(preprocessor,SVC(kernel='linear'))

    KNN=make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=2,weights='uniform',leaf_size=30,algorithm='auto'))
    Tree=make_pipeline(preprocessor,DecisionTreeClassifier(max_features=5,max_depth=4, random_state=0,min_samples_split=5))#,min_samples_split=5
    Extratree=make_pipeline(preprocessor,ExtraTreesClassifier(max_features=5,random_state=0,max_depth=4))

    RNA=make_pipeline(preprocessor, MLPClassifier(solver='lbfgs',hidden_layer_sizes=(8,),random_state=0,max_iter=1000))
    RForest=make_pipeline(preprocessor,RandomForestClassifier(random_state=0))
    GBoosting=make_pipeline(preprocessor,GradientBoostingClassifier(random_state=0))
    AdaBoost=make_pipeline(preprocessor, AdaBoostClassifier(random_state=0,n_estimators=40,learning_rate=1.0))
    Gaussian=make_pipeline(preprocessor,GaussianProcessClassifier(kernel=kernel_gp,n_restarts_optimizer=0))#1e-20
    Bayes=make_pipeline(preprocessor, GaussianNB())

app = Flask(__name__)


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/')
def index():
    return render_template('accueil.html')

@app.route('/accueil')
def accueil():
    return render_template('accueil.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/resultat')
def resultat():
    return render_template('resultat.html')

@app.route('/import')
def importation():
    return render_template('import.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/apropos')
def apropos():
    return render_template('apropos.html')

@app.route('/historiques')
def historiques():
    return render_template('historiques.html')

#login dans la page d'accueil
@app.route('/import', methods=['GET', 'POST'])
def connecter():

    error = None
    """
    res= c.execute("SELECT * from login where username=? and password=?",[])
    userrow = res.fetchone()

    username = userrow[0] # or whatever the index position is
    pwd=userrow[1]
    """

    if request.method == 'POST':
        if request.form['username'] != 'Admin' or request.form['password'] != 'root':
            error = '  Mot de passe incorret! Reessayer!'
        else:
            #return render_template('accueil.html', user=request.form['username'])
            return render_template('import.html')

    return render_template('index.html', error=error)
##Encodage data

def encodage(df):
    for col in df.select_dtypes('object').columns:
        df.loc[:,col]=df[col].astype('category').cat.codes
    return df

def obtenir_prevision(y,max):
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(3, 1, 1),
                                seasonal_order=(2, 1, 1, 5),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()

    #print(results.summary().tables[1])
    pred_uc = results.get_forecast(steps=max)

    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()
    prevision=pred_uc.predicted_mean

    return prevision

#Modelisation de serie temporelle
def serieTemporelle(data):
    from statsmodels.tsa.tsatools import lagmat
    from statsmodels.tsa.arima_model import ARIMA

    global df
    df=data.copy()
    
    y=df[type_target]

    # faire ne differenciation
    df["diff"] = np.nan
    df.loc[1:, "diff"] = (df.iloc[1:, 1].values - df.iloc[:len(df)-1, 1].values)

    global derniere_val,res_arima, forecasting

    derniere_val=df.iloc[len(df)-1:len(df), 1].values
    #model ARIMA

    forecasting=df[type_target].iloc[1:]
    try:
        arima_mod = ARIMA(forecasting, order=(1, 1, 1))
        res_arima = arima_mod.fit()
        #prevision_ts=res_arima.forecast(steps=0)[0]

        #print("Prevision forecasting : ",prevision_ts)

    except:
        print('Erreur', 'Erreur de forecasting!')
    
    #On créé la matrice avec les séries décalées.
    
    lag = 8
    X = lagmat(df["diff"], lag)
    lagged = df.copy()
    for c in range(1,lag+1):
        lagged["lag%d" % c] = X[:, c-1]

    #decoupe non aleatoire(serie temporelle) train/test
    xc = ["lag%d" % i for i in range(1,lag+1)]
    split = 0.66
    isplit = int(len(lagged) * split)
    xt = lagged[10:][xc]
    yt = lagged[10:]["diff"]

    X_train, y_train, X_test, y_test = xt[:isplit], yt[:isplit], xt[isplit:], yt[isplit:]


    return X_train,y_train,X_test,y_test

#Upload de fichier csv
@app.route('/dataset', methods=['GET', 'POST'])
def upload():
    global liste_colonne, data
    if request.method=='POST':
        fichier = request.files.get('file')

        #data=pd.read_csv(fichier)
        #extension=fichier.split('.')[-1]
        
        try:
            data=pd.read_csv(fichier)
        except:
            data=pd.read_excel(fichier)
        """
        if extension=='csv':
            data=pd.read_csv(fichier)
        else:
            data=pd.read_excel(fichier)
        """

        data=encodage(data)
        liste_colonne=list(data)
        shape=data.shape
        return render_template('dataset.html', shape=shape, liste_colonne=liste_colonne, fichier=fichier)

#Affichage des resultats
@app.route('/resultat', methods=['GET', 'POST'])
def predire():

    global liste_dataset, liste_entrainement

    liste_dataset=['Tabulaire','Série temporelle']
    liste_entrainement=['Régression','Classification']

    global X_train,X_test,y_train,y_test
    global type_dataset,type_entrainement,type_target,type_index

    if request.method=='POST':
        type_dataset = request.form.get('VariableIndexation')
        type_entrainement = request.form.get('typeEntrainement')
        type_index = request.form.get('Indexation')
        type_target = request.form.get('Target')

        X=data.drop(type_target, axis=1)
        y=data[type_target]

        if type_dataset==liste_dataset[0]:
            #creation de trainset et testset
            
            trainset, testset=train_test_split(data, test_size=0.2, random_state=2)
            
            X_train=trainset.drop(type_target, axis=1)
            y_train=trainset[type_target]

            X_test=testset.drop(type_target, axis=1)
            y_test=testset[type_target]

            #pred_rna, score_rna,t_rna, score_rna_val=evaluation(RNA)

            if type_entrainement==liste_entrainement[0]:

                regression()
            else:
                classification()

            def evaluation(model):
                global prediction
                t0=tm.time()
                model.fit(X_train,y_train)
                prediction=model.predict(X_test)
                #print(prediction)
                y_pred=prediction.mean()
                train_score=model.score(X_train,y_train)
                #train_score=train_scr.round(4)

                test_score=model.score(X_test,y_test)

                t1=tm.time()
                duree=t1-t0

                return y_pred, train_score, duree, test_score

            pred_ridge, score_ridge,t_ridge,score_ridge_val=evaluation(Ridge)
            pred_sgd, score_sgd,t_sgd, score_sgd_val=evaluation(SGD)
            pred_svm, score_svm,t_svm,score_svm_val=evaluation(SVM)
            pred_knn, score_knn,t_knn, score_knn_val=evaluation(KNN)
            pred_tree, score_tree,t_tree, score_tree_val=evaluation(Tree)
            pred_extratree, score_extratree,t_extratree, score_extratree_val=evaluation(Extratree)
            pred_rna, score_rna,t_rna, score_rna_val=evaluation(RNA)
            pred_rforest, score_rforest,t_rforest, score_rforest_val=evaluation(RForest)
            pred_gboosting, score_gboosting,t_gboosting, score_gboosting_val=evaluation(GBoosting)
            pred_adaboost, score_adaboost,t_adaboost, score_adaboost_val=evaluation(AdaBoost)
            pred_gaussian, score_gaussian,t_gaussian, score_gaussian_val=evaluation(Gaussian)


            global nom_model, liste_pred,score_mean,pred_mean

            nom_model=('Ridge','SGD','SVM','KNN','Tree','Extratree','RNA','RForest', 'GBoosting','AdaBoost','Gaussian')
            liste_pred=[pred_ridge,pred_sgd,pred_svm,pred_knn,pred_tree,pred_extratree,pred_rna,pred_rforest,pred_gboosting,pred_adaboost,pred_gaussian]
            
            list_score_train=[score_ridge,score_sgd,score_svm,score_knn,score_tree,score_extratree,score_rna,score_rforest,score_gboosting,score_adaboost,score_gaussian]
            list_score_val=[score_ridge_val,score_sgd_val,score_svm_val,score_knn_val,score_tree_val,score_extratree_val,score_rna_val,score_rforest_val,score_gboosting_val,score_adaboost_val,score_gaussian_val]
            list_duree=[t_ridge,t_sgd,t_svm,t_knn,t_tree,t_extratree,t_rna,t_rforest,t_gboosting,t_adaboost,t_gaussian]
            
            for i in range(len(list_score_train)):
                if list_score_train[i]<0:
                    list_score_train[i]=0.5
            for i in range(len(list_score_val)):
                if list_score_val[i]<0:
                    list_score_val[i]=0.5

            score_mean=stat.mean(list_score_val)
            pred_mean=stat.mean(liste_pred)
            y_mean=stat.mean(y)
            duree=stat.mean(list_duree)

            index=[]
            for i in range(len(y)):
                index.append(i)

            erreur=np.abs(y_mean-pred_mean)

            #SMA = y.rolling(window=5).mean()
            #SMA=data[type_target].ewm(span=40,adjust=False).mean()
            N = 2
            cumsum, SMA = [0], []

            for i, x in enumerate(y, 1):
                cumsum.append(cumsum[i-1] + x)
                if i>=N:
                    moving_ave = (cumsum[i] - cumsum[i-N])/N
                    #can do stuff with moving_ave here
                    SMA.append(moving_ave)
            SMA.append(pred_mean)


            train_sizes, train_score, val_score=learning_curve(RNA, X_train,y_train, cv=4, scoring='neg_mean_squared_error',train_sizes=np.linspace(0.1, 1, 16))
    
        else: #if serie temporelle

            X_train,y_train,X_test,y_test=serieTemporelle(data)

            regression()

            def evaluation(model):
                t0=tm.time()
                model.fit(X_train,y_train)
                prediction=model.predict(X_test)
                y_pred=prediction.mean().round(4)+derniere_val
                train_score=model.score(X_train,y_train)
                #train_score=train_scr.round(4)

                test_score=model.score(X_test,y_test)


                t1=tm.time()
                duree=t1-t0

                return y_pred, train_score, duree, test_score

            pred_ridge, score_ridge,t_ridge,score_ridge_val=evaluation(Ridge)
            pred_sgd, score_sgd,t_sgd, score_sgd_val=evaluation(SGD)
            pred_svm, score_svm,t_svm,score_svm_val=evaluation(SVM)
            pred_knn, score_knn,t_knn, score_knn_val=evaluation(KNN)
            pred_tree, score_tree,t_tree, score_tree_val=evaluation(Tree)
            pred_extratree, score_extratree,t_extratree, score_extratree_val=evaluation(Extratree)
            pred_rna, score_rna,t_rna, score_rna_val=evaluation(RNA)
            pred_rforest, score_rforest,t_rforest, score_rforest_val=evaluation(RForest)
            pred_gboosting, score_gboosting,t_gboosting, score_gboosting_val=evaluation(GBoosting)
            pred_adaboost, score_adaboost,t_adaboost, score_adaboost_val=evaluation(AdaBoost)
            pred_gaussian, score_gaussian,t_gaussian, score_gaussian_val=evaluation(Gaussian)


            nom_model=('Ridge','SGD','SVM','KNN','Tree','Extratree','RNA','RForest', 'GBoosting','AdaBoost','Gaussian')
            liste_pred=[pred_ridge[0],pred_sgd[0],pred_svm[0],pred_knn[0],pred_tree[0],pred_extratree[0],pred_rna[0],pred_rforest[0],pred_gboosting[0],pred_adaboost[0],pred_gaussian[0]]
            list_score_train=[score_ridge,score_sgd,score_svm,score_knn,score_tree,score_extratree,score_rna,score_rforest,score_gboosting,score_adaboost,score_gaussian]
            list_score_val=[score_ridge_val,score_sgd_val,score_svm_val,score_knn_val,score_tree_val,score_extratree_val,score_rna_val,score_rforest_val,score_gboosting_val,score_adaboost_val,score_gaussian_val]
            list_duree=[t_ridge,t_sgd,t_svm,t_knn,t_tree,t_extratree,t_rna,t_rforest,t_gboosting,t_adaboost,t_gaussian]
            
            for i in range(len(list_score_train)):
                if list_score_train[i]<0:
                    list_score_train[i]=0
            for i in range(len(list_score_val)):
                if list_score_val[i]<0:
                    list_score_val[i]=0

            score_mean=stat.mean(list_score_val)
            pred_mean=stat.mean(liste_pred)
            y_mean=stat.mean(y)
            duree=stat.mean(list_duree)

            erreur=np.abs(y_mean-pred_mean)

            train_sizes, train_score, val_score=learning_curve(RNA, X_train,y_train, cv=4, scoring='neg_mean_squared_error',train_sizes=np.linspace(0.1, 1, 16))

            prev=obtenir_prevision(y,100)
            SMA=list(y)+list(prev)

            index=[]
            for i in range(len(SMA)):
                index.append(i)

        return render_template('resultat.html',duree=duree,erreur=erreur.round(2), prediction=pred_mean.round(2), score_val=100*score_mean.round(4), liste_pred=liste_pred, labels=nom_model,train_score=train_score,val_score=val_score, train_sizes=train_sizes,list_score_train=list_score_train,list_score_val=list_score_val,prevision=list(SMA), index=index,y_reelle=list(y), cible=type_target)


if __name__ == '__main__':
    app.run(debug=True)