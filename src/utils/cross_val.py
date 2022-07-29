from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scikitplot as skplt
from tqdm import tqdm

from src.configs import config

class EST:
    def __init__(self, est):
        self.est = est
        self._estimator_type = "classifier"

    def predict_proba(self, X):
        return self.est.predict(X)


class CrossVal:
    def __init__(self, datasets, metas):
        self._datasets = shuffle(datasets, random_state=42)
        self._metas = shuffle(metas, random_state=42)
        self.split()
       
    def split(self):
        tumoral_idxs = self._metas
        normal_idxs = ~self._metas
        self.normal = self._datasets[normal_idxs]
        self.tumoral = self._datasets[tumoral_idxs]
        kfolds = StratifiedKFold(config.KFOLDS)
        self._split = kfolds.split(self._datasets, self._metas)

    def cross_val(self, model_init):
        fig, ax = plt.subplots(figsize=(15,10))

        tprs = list()
        aucs = list()
        models = list()
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train_idxs, test_idxs) in tqdm(enumerate(self._split)):
            models.append((
                model_init(), train_idxs, test_idxs))
            train, test = self._datasets.iloc[train_idxs], self._datasets.iloc[test_idxs]
            train_meta, test_meta = self._metas.iloc[train_idxs], self._metas.iloc[test_idxs]

            models[-1][0].fit(train, train_meta)
            predicted = models[-1][0].predict(test)
            fscore = f1_score(
                test_meta, predicted)
            viz = plot_roc_curve(
                models[-1][0], test, test_meta,
                name='ROC fold {}, f1: %0.2f'.format(i) % fscore, alpha=0.3, lw=1, ax=ax)
            
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic")
        ax.legend(loc="lower right")
        plt.show()
       
        return models

    def cross_val_sm(self):
        fig, ax = plt.subplots(figsize=(15,10))

        tprs = list()
        aucs = list()
        models = list()
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train_idxs, test_idxs) in tqdm(enumerate(self._split)):
            train, test = self._datasets.iloc[train_idxs], self._datasets.iloc[test_idxs]
            train_meta, test_meta = self._metas.iloc[train_idxs], self._metas.iloc[test_idxs]

            X2 = sm.add_constant(train)
            est = sm.Logit(train_meta, X2)
            est2 = est.fit()
            X3 = sm.add_constant(test)
            predicted = est2.predict(X3) > .5
            est2 = EST(est2)
            fscore = f1_score(
                test_meta, predicted)
            skplt.metrics.plot_roc_curve(test_meta, predicted, title='ROC fold {}, f1: {}'.format(i, fscore), ax=ax)
            
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic")
        ax.legend(loc="lower right")
        plt.show()
       
        return models
