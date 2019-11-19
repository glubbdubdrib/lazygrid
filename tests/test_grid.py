import unittest


class TestGrid(unittest.TestCase):
    
    def test_generate_grid(self):

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import RobustScaler, StandardScaler
        from sklearn.pipeline import Pipeline
        import lazygrid as lg

        preprocessors = [StandardScaler(), RobustScaler()]
        feature_selectors = [SelectKBest(score_func=f_classif, k=1), SelectKBest(score_func=f_classif, k=2)]
        classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]

        elements = [preprocessors, feature_selectors, classifiers]

        pipelines = lg.grid.generate_grid(elements)

        for pipeline in pipelines:
            self.assertTrue(isinstance(pipeline, Pipeline))
    
    def test_generate_grid_search(self):
        
        import keras
        from keras import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        import lazygrid as lg
        from keras.wrappers.scikit_learn import KerasClassifier

        # define keras model generator
        def create_keras_model(input_shape, optimizer, n_classes):
            kmodel = Sequential()
            kmodel.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation = 'relu', input_shape = input_shape))
            kmodel.add(MaxPooling2D(pool_size=(2, 2)))
            kmodel.add(Flatten())
            kmodel.add(Dense(1000, activation='relu'))
            kmodel.add(Dense(n_classes, activation='softmax'))

            kmodel.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optimizer, metrics=['accuracy'])

            return kmodel

        # cast keras model into sklearn model
        kmodel = KerasClassifier(create_keras_model)

        # define all possible model parameters of the grid
        model_params = {"optimizer": ['SGD', 'RMSprop'], "input_shape": [(28, 28, 3)], "n_classes": [10]}
        fit_params = {"epochs": [5, 10, 20], "batch_size": [10, 20]}

        # generate all possible models given the parameters' grid
        models, fit_parameters = lg.grid.generate_grid_search(kmodel, model_params, fit_params)

        # given the parameters, there are 12 possible combinations
        self.assertEqual(len(models), 12)
        self.assertEqual(len(fit_parameters), 12)

        # check model parameters
        for model in models:

            # fixed
            self.assertTrue(isinstance(model, Sequential))
            self.assertEqual(model.input_shape, (None, 28, 28, 3))
            self.assertEqual(model.output_shape, (None, 10))

            # variable
            self.assertTrue(model.optimizer.__class__.__name__ in ["SGD", "RMSprop"])

        # check fit parameters
        for fp in fit_parameters:
            self.assertTrue(fp["epochs"] in [5, 10, 20])
            self.assertTrue(fp["batch_size"] in [10, 20])


suite = unittest.TestLoader().loadTestsFromTestCase(TestGrid)
unittest.TextTestRunner(verbosity=2).run(suite)
