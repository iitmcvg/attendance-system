import os
import pickle
import numpy as np
from sklearn import neighbors, svm

BASE_DIR = os.path.dirname(__file__) + '/'
PATH_TO_PKL = 'trained_classifier.pkl'

import tensorflow as tf


class FaceClassifier(object):
    def __init__(self, model_path=None):

        self.model = None
        if model_path is None:
            return
        elif model_path == 'default':
            model_path = BASE_DIR+PATH_TO_PKL

        # Load models
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def my_rmse(self,labels, predictions):
        pred_values = predictions['predictions']
        return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

    def train(self, X, y, model='knn', save_model_path=None):
        if model in ['knn','SVM']:
            if model == 'knn':
                self.model = neighbors.KNeighborsClassifier(3, weights='uniform')
            elif model=="SVM":  # svm
                self.model = svm.SVC(kernel='linear', probability=True)
            
            self.model.fit(X, y)
            if save_model_path is not None:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(self.model, f)

        else:
            # Specify that all features have real-value data
            feature_columns = [tf.feature_column.numeric_column("X", shape=[512])]

            EVAL_INTERVAL = 300 # seconds
            run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                                keep_checkpoint_max = 3)


            # Build 3 layer DNN with 10, 20, 10 units respectively.
            classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=[256, 128],
                                                    n_classes=len(X),
                                                    model_dir="NN_model",
                                                    config = run_config)
            classifier = tf.contrib.estimator.add_metrics(classifier, my_rmse)
            # Define the training inputs
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"X": np.array(X)},
                y=np.arange(len(y)),
                num_epochs=None,
                shuffle=True)

            # Train model.

            classifier = tf.contrib.estimator.add_metrics(classifier, self.my_rmse)

            train_spec = tf.estimator.TrainSpec(
            input_fn = train_input_fn,
            max_steps = 2000)

            eval_spec = tf.estimator.EvalSpec(
            input_fn =  train_input_fn,  # no need to batch in eval
            steps = 200,
            start_delay_secs = 60, # start evaluating after N seconds
            throttle_secs = EVAL_INTERVAL)

            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

            accuracy_score = classifier.evaluate(input_fn=train_input_fn)["accuracy"]

            print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


    def classify(self, descriptor,model_type="NN"):
        if self.model is None:
            print('Train the model before doing classifications.')
            return
        
        if model_type.lower() in ['knn','svm']:
            print("sklearn model", self.model.predict([descriptor]),self.model.predict_proba([descriptor]))
            return self.model.predict([descriptor])[0],self.model.predict_proba([descriptor])[0]

        else:
            pass
