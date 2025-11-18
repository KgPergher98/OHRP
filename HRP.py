import pandas
import numpy
import riskfolio as rp

class HRP():

    def __init__(self, codep:str = "pearson", linkage:str = "ward"):
        self._options = {
            "codependence": codep,
            "linkage"     : linkage
        }

    def run(self, X):
        self.X = X.copy()
        numpy.random.seed(0)
        port = rp.HCPortfolio(returns = self.X)
        weights_hrp = port.optimization(
            model = "HRP",
            codependence = self._options["codependence"],
            rm = "MV",
            rf = 0.0,
            linkage = self._options["linkage"],
            max_k = 40,
            leaf_order = True
        )
        weights_hrp = weights_hrp.weights
        weights_hrp /= numpy.sum(weights_hrp)
        return weights_hrp