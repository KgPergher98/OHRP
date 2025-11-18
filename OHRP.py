import pandas
import numpy
import riskfolio as rp

from COLPP import COLPP

class OHRP():

    def __init__(self, codep:str = "pearson", linkage:str = "ward") -> None:
        self._options = { # COMO QUEREMOS CLUSTERIZAR
            "gnd"         : None,
            "NeighborMode": "Supervised", # SUPERVISIONADO (COLPP)
            "WeightMode"  : "HeatKernel",   # HEAT KERNEL
            "bNormalized" : 0,
            "bLDA"        : 0,
            "codependence": codep,
            "linkage"     : linkage
        }

    def _convertLabel(lbs):
        conv = pandas.DataFrame(lbs.label.unique().tolist(), columns = ["setor"])
        conv["idx"] = conv.index
        conv.index = conv.setor
        lbs2 = lbs.copy()
        for t in range(lbs2.shape[0]):
            lbs2.iloc[t, 0] = conv.loc[lbs2.iloc[t, 0], "idx"]
        return conv, lbs2
    
    def run(self, X, k:list = [5], d:list = [10], r:list = [0.9], L = None):

        self.X = X.copy()
        self._options["t"] = COLPP.find_optimal(df = self.X)
        self.X = self.X.transpose()

        # Adjust Labels
        if L == None:
            L = pandas.DataFrame("X", index = X.columns.tolist(), columns = ["label"])
        conv, self.L = OHRP._convertLabel(lbs = L)

        # Adjust iterables
        if type(k) == int  : k = [k]
        if type(d) == int  : d = [d]
        if type(r) == float: r = [r]

        best_volatility = - numpy.inf
        best_combo      = {"d": 0, "k": 0, "r": 0}
        best_weights    = pandas.Series(numpy.nan, index = self.X.columns.tolist(), name = "weights")

        for ki in k:
            for di in d:
                for ri in r:
                    #try:
                    weights, inSampleRet = self.OrthoRedDim(
                        X = self.X.copy(), L = self.L.copy(), k = ki, d = di, r = ri
                    )
                    vol = 1/numpy.std(inSampleRet)
                    if vol > best_volatility:
                        best_volatility = vol
                        best_combo["d"] = di
                        best_combo["k"] = ki
                        best_combo["r"] = ri
                        best_weights    = weights.copy()
                    #except Exception as exc:
                    #    print(f"Error occurred: {exc}")

        self._bestHyperparams = best_combo
        
        return best_weights


    def OrthoRedDim(self, X, L, k:int, d:int, r:float):

        self._options["k"]          = k
        self._options["ReducedDim"] = d
        self._options["PCARatio"]   = r

        # AFINIDADE
        W = COLPP.affinity_matrix(df = X, labels = L, ops = self._options, self_connection = False, olpp = True)

        D = COLPP.diagonal(affinity = W)

        X_std = (X.copy() - X.copy().mean(axis = 0))
        # DECOMPOSICAO SVD
        U, S, V = COLPP.SVD(X = X_std)

        U, S, V = COLPP.cut_on_ratio(U = U, V = V, S = S, pca_ratio = self._options["PCARatio"])

        # ESPAÇO ORTOGONAL
        eigen_vector = COLPP.build_weights(
            U = U, S = S, V = V, D = D, W = W, reduced_dim = self._options["ReducedDim"], bd = True
        )

        # PROJETA OS RETORNOS
        red_space = COLPP.project_data(X = X, P = eigen_vector, L = L)
        red_space = red_space[red_space.drop(["label"], axis = 1).std(axis = 1) != 0]

        #numpy.random.seed(0)
        port = rp.HCPortfolio(returns = red_space.drop(["label"], axis = 1).transpose().copy())
        weights_ohrp = port.optimization(
            model = "HRP",
            codependence = self._options["codependence"],
            rm = "MV",
            rf = 0.0,
            linkage = self._options["linkage"],
            max_k = 40,
            leaf_order = True
        )

        # NORMALIZA PARA PESO 1 (DESCONSIDERA PESOS MUITO PEQUENOS)
        #weights_ohrp = weights_ohrp[weights_ohrp.weights > 0.005].weights
        weights_ohrp = weights_ohrp.weights
        weights_ohrp /= numpy.sum(weights_ohrp)
        # OUT-OF-SAMPLE
        #out = Y.copy() * weights_ohrp.transpose()
        #out.dropna(axis = 1, how = "all", inplace = True)
        #out = out.sum(axis = 1)
        # IN-SAMPLE
        inp = X.transpose().copy() * weights_ohrp.transpose()
        inp.dropna(axis = 1, how = "all", inplace = True)
        inp = inp.sum(axis = 1)

        #weights_ohrp = pandas.DataFrame(weights_ohrp).transpose()
        #weights_ohrp.index = [out.index.tolist()[0]]
        return weights_ohrp, inp