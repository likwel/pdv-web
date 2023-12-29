
def regression():

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

	global Ridge,SGD,SVM,KNN,Tree,Extratree,RNA,RForest,GBoosting,AdaBoost,Gaussian

	import matplotlib.pyplot as plt

	kernel_gp = DotProduct() + WhiteKernel()

	preprocessor=make_pipeline(PolynomialFeatures(2, include_bias=False),StandardScaler(), SelectKBest(f_classif,k='all'))

	Ridge=make_pipeline(preprocessor, Ridge(random_state=0))
	SGD=make_pipeline(preprocessor,SGDRegressor(loss='squared_loss',random_state=0,max_iter=1000, tol=1e-3))
	SVM=make_pipeline(preprocessor,SVR(kernel='linear'))

	KNN=make_pipeline(preprocessor, KNeighborsRegressor(n_neighbors=2,weights='uniform',leaf_size=30,algorithm='auto'))
	Tree=make_pipeline(preprocessor,DecisionTreeRegressor(max_features=5,max_depth=4, random_state=0,min_samples_split=5))#,min_samples_split=5
	Extratree=make_pipeline(preprocessor,ExtraTreesRegressor(max_features=5,random_state=0,max_depth=4))

	RNA=make_pipeline(preprocessor, MLPRegressor(solver='lbfgs',hidden_layer_sizes=(8,),random_state=0,max_iter=1000))
	RForest=make_pipeline(preprocessor,RandomForestRegressor(random_state=0))
	GBoosting=make_pipeline(preprocessor,GradientBoostingRegressor(random_state=0))
	AdaBoost=make_pipeline(preprocessor, AdaBoostRegressor(random_state=0,n_estimators=40, loss='linear',learning_rate=1.0))
	Gaussian=make_pipeline(preprocessor,GaussianProcessRegressor(kernel=kernel_gp,n_restarts_optimizer=0, normalize_y=True, alpha=.5))#1e-20

	global liste_pred, nom_model, list_score_train, list_score_val
	global score_ridge,score_sgd,score_svm,score_knn,score_tree,score_extratree,score_rna,score_rforest,score_gboosting,score_adaboost,score_gaussian
	global t_ridge,t_sgd,t_svm,t_knn,t_tree,t_extratree,t_rna,t_rforest,t_gboosting,t_adaboost,t_gaussian
	global pred_ridge,pred_sgd,pred_svm,pred_knn,pred_tree,pred_extratree,pred_rna,pred_rforest,pred_gboosting,pred_adaboost,pred_gaussian


	def evaluation(model):
		t0=tm.time()
		model.fit(X_train,y_train)
		prediction=model.predict(X_test)
		#print(prediction)
		y_pred=prediction.mean().round(4)
		train_score=model.score(X_train,y_train).round(4)
		#train_score=train_scr.round(4)

		test_score=model.score(X_test,y_test).round(4)

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
	liste_pred=[pred_ridge,pred_sgd,pred_svm,pred_knn,pred_tree,pred_extratree,pred_rna,pred_rforest,pred_gboosting,pred_adaboost,pred_gaussian]
	
	list_score_train=[score_ridge,score_sgd,score_svm,score_knn,score_tree,score_extratree,score_rna,score_rforest,score_gboosting,score_adaboost,score_gaussian]
	list_score_val=[score_ridge_val,score_sgd_val,score_svm_val,score_knn_val,score_tree_val,score_extratree_val,score_rna_val,score_rforest_val,score_gboosting_val,score_adaboost_val,score_gaussian_val]
	global score_mean
	score_mean=list_score_val.mean()

