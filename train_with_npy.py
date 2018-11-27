import numpy as np
from classifier.FaceClassifier import  FaceClassifier
#Training  classifiers after obtaining the feaure vectors from the recognition model
features=np.load('./npy files/features.npy')
labels=np.load('./npy files/labels.npy')
face_classifier = FaceClassifier('./classifier/trained_svm.pkl')
Svm = face_classifier.train(features,labels,model='SVM',save_model_path='./classifier/new_classifiers/trained_svm.pkl')
#KNN = face_classifier.train(features,labels,model='knn',save_model_path='./classifier/trained_classifier.pkl')
KNN_5 = face_classifier.train(features,labels,model='knn',knn_neighbours=5,save_model_path='./classifier/new_classifiers/knn_5.pkl')
KNN_7 = face_classifier.train(features,labels,model='knn',knn_neighbours=7,save_model_path='./classifier/new_classifiers/knn_7.pkl')
random_forests= face_classifier.train(features,labels,model='random-forests',save_model_path='./classifier/new_classifiers/random_forests.pkl')
Svm_poly = face_classifier.train(features,labels,model='SVC-poly',save_model_path='./classifier/new_classifiers/trained_svm_poly.pkl')
Svm_rbf = face_classifier.train(features,labels,model='SVC-rbf',save_model_path='./classifier/new_classifiers/trained_svm_rbf.pkl')
