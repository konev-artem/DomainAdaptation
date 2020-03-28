class BaseModel:
    def call(self, inputs, training=None):
        raise NotImplementedError
