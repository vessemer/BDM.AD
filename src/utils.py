import pandas as pd
import numpy as np


def get_qvals(names, is_np_header=False):
    template = '../data/intersection/{pref}{name}.bed'
    header = [ 'chr', 'start', 'end', 'id', 'qval' ]
    np_header = [ 
        'chrom', 'chromStart', 'chromEnd', 'name', 
        'score', 'strand', 'signalValue', 'pValue', 'qval', 'peak' ]
    qvals = list()

    for name in names:
        path = template.format(pref='{}', name=name)
        intersected = pd.read_csv(path.format(''), sep='\t', header=None)
        intersected.columns = header + np.arange(intersected.shape[1] - len(header)).tolist()
        non_intersected = pd.read_csv(path.format('NON_'), sep='\t', header=None)
        non_intersected.columns = np_header if is_np_header else header

        qvals.append(pd.DataFrame({ 
            'qval': np.concatenate([intersected.qval.values, non_intersected.qval.values]),
            'is_intersected': np.concatenate([np.ones_like(intersected.qval), np.zeros_like(non_intersected.qval)]).astype(np.bool_),
            'name': [name] * (len(intersected.qval) + len(non_intersected.qval))
        }))

    qvals = pd.concat(qvals)
    qvals['log_qval'] = np.log(qvals.qval)
    return qvals
