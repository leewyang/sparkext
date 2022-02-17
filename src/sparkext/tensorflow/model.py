import tensorflow as tf
from sparkext.model import BaseModel

class TFModel(BaseModel):
    def __init__(self, model):
        self.model = None
        super(TFModel, self).__init__(model)

    def _from_file(self, model_path):
        self.model = tf.saved_model.load(model_path)
        # self.model = tf.keras.models.load_model(model_path)

    def _from_object(self, model):
        self.model = model

    def _transform(self, dataset):
        def predict_udf(batch_iter):
            predictions = self.model.predict()

