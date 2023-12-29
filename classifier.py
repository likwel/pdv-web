def classification():


	global Ridge,SGD,SVM,KNN,Tree,Extratree,RNA,RForest,GBoosting,AdaBoost,Gaussian, Bayes

	import matplotlib.pyplot as plt

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

	from time import time

	import matplotlib.pyplot as plt

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

	def evaluation(model):
		global prediction
		t0=tm.time()
		model.fit(X_train,y_train)

		prediction=model.predict(X_test)
		#prediction=model.predict_proba(X_test)
		y_pred=prediction.mean().round(4)
		train_score=model.score(X_train,y_train).round(4)
		#train_score=train_scr.round(4)

		test_score=model.score(X_test,y_test).round(4)

		t1=tm.time()
		duree=t1-t0
		
		Acc=accuracy_score(y_test, prediction)
		F1=f1_score(y_test, prediction,average='weighted')
		Rec=recall_score(y_test, prediction,average='weighted')
		prec=precision_score(y_test, prediction,average='weighted')
		
		global mat_conf
		mat_conf=confusion_matrix(y_test, prediction)

		global dict_metric
		dict_metric={'Acc': Acc,'F1':F1,'Rec':Rec,'Prec':prec}

		
		print('Prediction : ',y_pred.mean())
		print('Score train : ',train_score)

		return y_pred, train_score, duree, test_score

	global score_ridge,score_sgd,score_svm,score_knn,score_tree,score_extratree,score_rna,score_rforest,score_gboosting,score_adaboost,score_gaussian, score_bayes
	global t_ridge,t_sgd,t_svm,t_knn,t_tree,t_extratree,t_rna,t_rforest,t_gboosting,t_adaboost,t_gaussian, t_bayes
	global pred_ridge,pred_sgd,pred_svm,pred_knn,pred_tree,pred_extratree,pred_rna,pred_rforest,pred_gboosting,pred_adaboost,pred_gaussian, pred_bayes

	print("Ridge :")
	pred_ridge, score_ridge,t_ridge,score_ridge_val=evaluation(Ridge)
	print("SGDRegressor :")
	pred_sgd, score_sgd,t_sgd, score_sgd_val=evaluation(SGD)
	print("SVM :")
	pred_svm, score_svm,t_svm,score_svm_val=evaluation(SVM)
	print("KNN :")
	pred_knn, score_knn,t_knn, score_knn_val=evaluation(KNN)
	print("Tree :")
	pred_tree, score_tree,t_tree, score_tree_val=evaluation(Tree)
	print("Extratree :")
	pred_extratree, score_extratree,t_extratree, score_extratree_val=evaluation(Extratree)
	print("RNA :")
	pred_rna, score_rna,t_rna, score_rna_val=evaluation(RNA)
	print("RandomForest:")
	pred_rforest, score_rforest,t_rforest, score_rforest_val=evaluation(RForest)
	print("GBoosting :")
	pred_gboosting, score_gboosting,t_gboosting, score_gboosting_val=evaluation(GBoosting)
	print("AdaBoost : ")
	pred_adaboost, score_adaboost,t_adaboost, score_adaboost_val=evaluation(AdaBoost)
	print("Gaussian : ")
	pred_gaussian, score_gaussian,t_gaussian, score_gaussian_val=evaluation(Gaussian)
	print("Bayes : ")
	pred_bayes, score_bayes,t_bayes, score_bayes_val=evaluation(Bayes)

	global liste_pred, nom_model, list_score_train, list_score_val

	nom_model=('Ridge','SGD','SVM','KNN','Tree','Extratree','RNA','RForest', 'GBoosting','AdaBoost','Gaussian', 'NBayes')
	liste_pred=[pred_ridge,pred_sgd,pred_svm,pred_knn,pred_tree,pred_extratree,pred_rna,pred_rforest,pred_gboosting,pred_adaboost,pred_gaussian, pred_bayes]
	list_score_train=[score_ridge,score_sgd,score_svm,score_knn,score_tree,score_extratree,score_rna,score_rforest,score_gboosting,score_adaboost,score_gaussian,score_bayes]
	list_score_val=[score_ridge_val,score_sgd_val,score_svm_val,score_knn_val,score_tree_val,score_extratree_val,score_rna_val,score_rforest_val,score_gboosting_val,score_adaboost_val,score_gaussian_val,score_bayes_val]
