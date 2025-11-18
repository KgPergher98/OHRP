import pandas

class EW():
    def __init__(self):
        pass

    def run(self, X):
        self.X = X.copy()
        weights_ew = pandas.DataFrame(1 / self.X.shape[1], index = self.X.columns.tolist(), columns = ["weights"])
        return weights_ew.weights