import argparse

import numpy as np
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.models.classification.vanilla_classifier import TMClassifier

from ytm.utils import Profiler

from ..data import load_mnist
from ..log import RunLog
from .pysparse import PySparseTM, to_csr


def prepare(X_train, Y_train, X_test, Y_test, flat: bool):
    if flat:
        X_train, X_test = X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1)
    return X_train.astype(np.uint32), Y_train.astype(np.uint32), X_test.astype(np.uint32), Y_test.astype(np.uint32)


def train_model(tm, X_train, Y_train, X_test, Y_test, epochs: int, log: RunLog, expname: str):
    for epoch in range(1, epochs + 1):
        with Profiler() as fit_p:
            tm.fit(X_train, Y_train)
        with Profiler() as pred_p:
            pred = tm.predict(X_test)
        fit, predict = fit_p.get_profile(), pred_p.get_profile()
        acc = (pred == Y_test).mean()
        log.write(epoch, fit, predict, {"acc": acc})
        print(f"{expname} epoch {epoch}/{epochs}  fit {fit.elapsed:.2f}s  predict {predict.elapsed:.2f}s  acc {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=lambda v: None if v.lower() == "none" else (int(v), int(v)), default="10")
    parser.add_argument("--coalesced", type=int, choices=(0, 1), default=1)
    parser.add_argument("--n_clauses", type=int, default=500)
    parser.add_argument("--T", type=float, default=1000)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10])
    parser.add_argument("--platform", default="CPU")
    args = parser.parse_args()

    expname = "mnist_conv" if args.patch else "mnist_flat"

    pysparse = args.coalesced and args.platform == "CUDA"
    log = RunLog("pysparse" if pysparse else "tmu", expname, {**vars(args), "weighted_clauses": True})

    with Profiler() as setup_p:
        data = prepare(*load_mnist(), flat=args.patch is None)
        if pysparse:
            data = to_csr(*data)
    log.set_setup(setup_p.get_profile())

    for seed in args.seeds:
        print(f"{expname} seed {seed}")
        with Profiler() as init_p:
            if pysparse:
                from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D, MultiClassTsetlinMachine

                if args.patch:
                    tm = PySparseTM(MultiClassConvolutionalTsetlinMachine2D(args.n_clauses, args.T, args.s, (28, 28, 1), args.patch))
                else:
                    tm = PySparseTM(MultiClassTsetlinMachine(args.n_clauses, args.T, args.s))
            else:
                TM = TMCoalescedClassifier if args.coalesced else TMClassifier
                tm = TM(args.n_clauses, args.T, args.s, platform=args.platform, patch_dim=args.patch, weighted_clauses=True, seed=seed)
        log.start(seed, init_p.get_profile())
        train_model(tm, *data, epochs=args.epochs, log=log, expname=expname)
