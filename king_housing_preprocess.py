import torch
import arff
import pandas as pd


# Preprocessing code is from "Conditionally Risk-Averse Contextual Bandits" [Farsang et al 2022]
# Paper Link: https://arxiv.org/pdf/2210.13573.pdf


# Note: paths to datasets and files are removed
def arff_to_df():
    # fill in with path to .arff dataset file
    data = arff.load(open("", "r"))
    z = pd.DataFrame(data["data"])
    z.columns = [v[0].lower() for v in data["attributes"]]
    return z


class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sumsq = 0

    def __iadd__(self, other):
        import math

        if not math.isnan(other):
            self.n += 1
            self.sum += other
            self.sumsq += other * other
        return self

    def __isub__(self, other):
        import math

        if not math.isnan(other):
            self.n += 1
            self.sum -= other
            self.sumsq += other * other
        return self

    def mean(self):
        return self.sum / max(self.n, 1)

    def var(self):
        from math import sqrt

        return sqrt(self.sumsq / max(self.n, 1) - self.mean() ** 2)

    def semean(self):
        from math import sqrt

        return self.var() / sqrt(max(self.n, 1))


class EasyPoissonBootstrapAcc:
    def __init__(self, batch_size, confidence=0.95, seed=2112):
        from math import ceil
        from numpy.random import default_rng

        self.n = 0
        self.batch_size = batch_size
        self.confidence = confidence
        self.samples = [EasyAcc() for _ in range(int(ceil(3 / (1 - self.confidence))))]
        self.rng = default_rng(seed)

    def __iadd__(self, other):
        self.n += 1

        poissons = (
            self.rng.poisson(lam=self.batch_size, size=len(self.samples))
            / self.batch_size
        )

        for n, (chirp, acc) in enumerate(zip(poissons, self.samples)):
            acc += (chirp if n > 0 else 1) * other

        return self

    def __isub__(self, other):
        return self.__iadd__(-other)

    def ci(self):
        import numpy

        quantiles = numpy.quantile(
            a=[x.mean() for x in self.samples],
            q=[1 - self.confidence, 0.5, self.confidence],
        )
        return list(quantiles)

    def formatci(self):
        z = self.ci()
        return "[{:<.4f},{:<.4f}]".format(z[0], z[2])


class Schema(object):
    def __init__(self, *, attributes, target, skipcol, data):
        super().__init__()

        schema = {}
        n = 0
        for kraw, v in attributes:
            k = kraw.lower()

            if k in skipcol:
                continue

            if isinstance(v, str):
                if v in ["INTEGER", "REAL"]:
                    if any(
                        thisv is None
                        for row in data
                        for thisk, thisv in zip(attributes, row)
                        if thisk[0].lower() == k
                    ):
                        assert k != target, (k, target)
                        schema[k] = (
                            lambda i: (lambda z: (i + 1, 1) if z is None else (i, z))
                        )(n)
                        n += 2
                    else:
                        schema[k] = (lambda i: (lambda z: (i, z)))(n)
                        n += 1
                elif k == "date":
                    import ciso8601
                    import time

                    schema[k] = (
                        lambda i: (
                            lambda z: (
                                i,
                                time.mktime(ciso8601.parse_datetime(z).timetuple()),
                            )
                        )
                    )(n)
                    n += 1
                elif v == "STRING":
                    uniques = set(
                        [
                            thisv
                            for row in data
                            for thisk, thisv in zip(attributes, row)
                            if thisk[0].lower() == k
                        ]
                    )
                    schema[k] = (lambda h: (lambda z: (h[z], 1)))(
                        {z: (n + m) for m, z in enumerate(uniques)}
                    )
                    n += len(uniques)
                else:
                    assert False, (k, v)
            elif isinstance(v, list) and all((isinstance(z, str) for z in v)):
                assert k != target, (k, target)
                schema[k] = (lambda h: (lambda z: (h[z], 1)))(
                    {z: (n + m) for m, z in enumerate(v)}
                )
                n += len(v)
            else:
                assert False

            if k == target:
                n -= 1

        assert target in schema, (target, attributes)

        self.schema = schema
        self.target = target
        self.nfeatures = n

    def featurize(self, colname, val):
        if colname in self.schema:
            yield self.schema[colname](val)


def makeData(filename, *, target, skipcol, skiprow):
    import arff
    import numpy

    data = arff.load(open(filename, "r"))
    schema = Schema(
        attributes=data["attributes"], target=target, skipcol=skipcol, data=data["data"]
    )

    Y = []
    X = []

    for row in data["data"]:
        hashrow = {kraw[0].lower(): v for kraw, v in zip(data["attributes"], row)}

        if skiprow(hashrow):
            continue

        y = None
        x = [0] * schema.nfeatures
        for col, val in hashrow.items():
            if col == target:
                y = next(schema.featurize(col, val))[1]
            else:
                for f, vf in schema.featurize(col, val):
                    from numbers import Number

                    assert isinstance(vf, Number), (col, val, f, vf)
                    x[f] = vf

        Y.append(y)
        X.append(x)

    Y = numpy.array(Y)
    Ymin, Ymax = numpy.min(Y), numpy.max(Y)
    Y = (Y - Ymin) / (Ymax - Ymin)
    X = numpy.array(X)
    Xmin, Xmax = numpy.min(X, axis=0, keepdims=True), numpy.max(
        X, axis=0, keepdims=True
    )
    if numpy.any(Xmin >= Xmax):
        X = X[:, Xmin[0, :] < Xmax[0, :]]
        Xmin, Xmax = numpy.min(X, axis=0, keepdims=True), numpy.max(
            X, axis=0, keepdims=True
        )
    assert numpy.all(Xmax > Xmin), [
        (col, lb, ub)
        for col, (lb, ub) in enumerate(zip(Xmin[0, :], Xmax[0, :]))
        if lb >= ub
    ]
    X = (X - Xmin) / (Xmax - Xmin)

    return X, Y


class ArffToPytorch(torch.utils.data.Dataset):
    def __init__(self, filename, *, target, skipcol, skiprow):
        X, Y = makeData(filename, target=target, skipcol=skipcol, skiprow=skiprow)
        self.Xs = torch.Tensor(X)
        self.Ys = torch.Tensor(Y).unsqueeze(1)

    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        # Select sample
        return self.Xs[index], self.Ys[index]
