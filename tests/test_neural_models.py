import unittest


class TestNeuralModels(unittest.TestCase):

    def test_keras_classifier(self):

        from keras import Sequential
        import lazygrid as lg

        keras_model = lg.neural_models.keras_classifier(layers=[10, 5], input_shape=(20,),
                                                        n_classes=4, verbose=True)
        keras_model.layers[0].trainable = False
        lg.neural_models.reset_weights(keras_model)

        self.assertTrue(isinstance(keras_model, Sequential))


suite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralModels)
unittest.TextTestRunner(verbosity=2).run(suite)
