import unittest


class TestModelSelection(unittest.TestCase):

    def test_cross_validation_sklearn_model(self):

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import lazygrid as lg

        x, y = make_classification(random_state=42)

        classifier = RandomForestClassifier(random_state=42)

        model = lg.wrapper.SklearnWrapper(classifier)
        score, fitted_models, y_pred_list, y_true_list = lg.model_selection.cross_validation(model=model, x=x, y=y)

        conf_mat = lg.plotter.generate_confusion_matrix(fitted_models[-1].model_id, fitted_models[-1].model_name,
                                                        y_pred_list, y_true_list, class_names={0: "N", 1: "P"})

        # check models' type
        for fitted_model in fitted_models:
            self.assertTrue(isinstance(fitted_model, lg.wrapper.SklearnWrapper))

        # check confusion matrix
        self.assertTrue(conf_mat.matrix == {"N": {"N": 48, "P": 2},
                                            "P": {"N": 5, "P": 45}})

    def test_cross_validation_generic_score(self):

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_breast_cancer
        import lazygrid as lg
        import numpy as np

        def compute_class_imbalance_ratio(y_val, *args, **kwargs):
            """
            Compute class-imbalance ratio of the validation set.
            """

            values, counts = np.unique(y_val, return_counts=True)
            pmax = np.max(counts)  # majority class
            pmin = np.min(counts)  # minority class
            imbalance_ratio = np.round(pmax / pmin, 4)
            return imbalance_ratio

        x, y = load_breast_cancer(return_X_y=True)

        classifier = RandomForestClassifier(random_state=42)

        model = lg.wrapper.SklearnWrapper(classifier)
        score, _, _, _ = lg.model_selection.cross_validation(model=model, x=x, y=y,
                                                             generic_score=compute_class_imbalance_ratio)

        self.assertTrue(score == {0: 1.6364, 1: 1.6364, 2: 1.7143, 3: 1.7143, 4: 1.7143,
                                  5: 1.7143, 6: 1.7143, 7: 1.6667, 8: 1.6667, 9: 1.6667})


    def test_cross_validation_sklearn_pipeline(self):

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.datasets import make_classification
        import lazygrid as lg

        x, y = make_classification(random_state=42)

        standardizer = StandardScaler()
        feature_selector = SelectKBest(score_func=mutual_info_classif, k=2)
        classifier = RandomForestClassifier(random_state=42)
        pipeline = Pipeline(steps=[("standardizer", standardizer),
                                   ("feature_selector", feature_selector),
                                   ("classifier", classifier)])

        db_name = "database-test"
        dataset_id = 1
        dataset_name = "make-classification"

        model = lg.wrapper.PipelineWrapper(pipeline, db_name=db_name, dataset_id=dataset_id, dataset_name=dataset_name)
        score, fitted_models, y_pred_list, y_true_list = lg.model_selection.cross_validation(model=model, x=x, y=y)

        conf_mat = lg.plotter.generate_confusion_matrix(fitted_models[-1].model_id, fitted_models[-1].model_name,
                                                y_pred_list, y_true_list)

        # check models' type
        for fitted_model in fitted_models:
            self.assertTrue(isinstance(fitted_model, lg.wrapper.PipelineWrapper))

        # check confusion matrix
        self.assertTrue(conf_mat.matrix == {0: {0: 46, 1: 4},
                                            1: {0: 3, 1: 47}})

    def test_cross_validation_keras_model(self):

        import keras
        from keras import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        from keras.utils import to_categorical
        from sklearn.metrics import f1_score
        from sklearn.datasets import load_digits
        from sklearn.model_selection import StratifiedKFold
        import lazygrid as lg
        import numpy as np
        import random as rn
        import os
        import tensorflow as tf

        seed = 42  # Here sd means seed.
        np.random.seed(seed)
        rn.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        from keras import backend as K
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=config)
        K.set_session(sess)

        # define keras model generator
        def create_keras_model(optimizer):

            kmodel = Sequential()
            kmodel.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                              activation='relu',
                              input_shape=x_train.shape[1:]))
            kmodel.add(MaxPooling2D(pool_size=(2, 2)))
            kmodel.add(Flatten())
            kmodel.add(Dense(1000, activation='relu'))
            kmodel.add(Dense(n_classes, activation='softmax'))

            kmodel.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optimizer,
                           metrics=['accuracy'])
            return kmodel

        # load data set
        x, y = load_digits(return_X_y=True)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        list_of_splits = [split for split in skf.split(x, y)]
        train_index, val_index = list_of_splits[0]
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        x_train = np.reshape(x_train, (x_train.shape[0], 8, 8, 1))
        x_val = np.reshape(x_val, (x_val.shape[0], 8, 8, 1))
        n_classes = len(np.unique(y_train))
        if n_classes > 2:
            y_train = to_categorical(y_train)
            y_val = to_categorical(y_val)

        # cast keras model into sklearn model
        kmodel = create_keras_model(optimizer="SGD")
        fit_params = {"epochs": 5, "batch_size": 10, "shuffle": False}

        # define scoring function for one-hot-encoded labels
        def score_fun(y, y_pred):
            y = np.argmax(y, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
            return f1_score(y, y_pred, average="weighted")

        db_name = "database-test"
        dataset_id = 2
        dataset_name = "digits"

        # cross validation
        model = lg.wrapper.KerasWrapper(kmodel, fit_params=fit_params, db_name=db_name,
                                        dataset_id=dataset_id, dataset_name=dataset_name)
        score, fitted_models, \
            y_pred_list, y_true_list = lg.model_selection.cross_validation(model=model, x=x_train, y=y_train,
                                                                           x_val=x_val, y_val=y_val,
                                                                           random_data=False, n_splits=3,
                                                                           score_fun=score_fun)

        conf_mat = lg.plotter.generate_confusion_matrix(fitted_models[-1].model_id, fitted_models[-1].model_name,
                                                        y_pred_list, y_true_list, encoding="one-hot")

        # check models' type
        for fitted_model in fitted_models:
            self.assertTrue(isinstance(fitted_model, lg.wrapper.KerasWrapper))

        # check confusion matrix
        self.assertTrue(conf_mat.matrix == {0: {0: 54, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 1: {0: 0, 1: 37, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 14, 9: 4}, 2: {0: 1, 1: 0, 2: 53, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 3: {0: 0, 1: 0, 2: 0, 3: 55, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 0}, 4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 57, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 5: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 54, 6: 0, 7: 0, 8: 0, 9: 3}, 6: {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 55, 7: 0, 8: 1, 9: 0}, 7: {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 51, 8: 0, 9: 2}, 8: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 52, 9: 1}, 9: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 54}})
        
    def test_compare_models(self):

        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import pandas as pd
        import lazygrid as lg

        dataset_id = 1
        dataset_name = "make-classification"
        db_name = "lazygrid-test"

        x, y = make_classification(random_state=42)

        lg_model_1 = lg.wrapper.SklearnWrapper(LogisticRegression(), dataset_id=dataset_id,
                                       dataset_name=dataset_name, db_name=db_name)
        lg_model_2 = lg.wrapper.SklearnWrapper(RandomForestClassifier(), dataset_id=dataset_id,
                                       dataset_name=dataset_name, db_name=db_name)
        lg_model_3 = lg.wrapper.SklearnWrapper(RidgeClassifier(), dataset_id=dataset_id,
                                       dataset_name=dataset_name, db_name=db_name)

        models = [lg_model_1, lg_model_2, lg_model_3]
        results = lg.model_selection.compare_models(models=models, x_train=x, y_train=y)

        results.to_html('results.html')

        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertEqual(len(results), 3)


suite = unittest.TestLoader().loadTestsFromTestCase(TestModelSelection)
unittest.TextTestRunner(verbosity=2).run(suite)
