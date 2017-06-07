#! /usr/bin/env python
import operator
from functools import reduce
import numpy as np
import pandas as pd
from itertools import chain


class Factor(object):
    def __init__(self, df):
        assert df.columns == ['phi']
        names = df.index.names
        self.values = df.reset_index().reindex_axis(sorted(chain(df.columns, names)), axis=1).set_index(names)

    @classmethod
    def from_scratch(cls, variables, variable_cardinalities, values=None):
        if not values:
            values = np.ones(np.prod(np.array(variable_cardinalities)))
        else:
            assert len(values) == reduce(operator.mul, variable_cardinalities)
        index = pd.MultiIndex.from_product([range(c) for c in variable_cardinalities], names=variables)
        df = pd.DataFrame(data=values, index=index, columns=['phi'])
        return cls(df)

    @property
    def variables(self):
        return self.values.index.names

    @property
    def variable_cardinalities(self):
        return list(map(len, self.values.index.levels))

    def normalize(self):
        self.values = self.values / self.values.sum()

    def __mul__(self, other):
        return factor_product(self, other)

    def __eq__(self, other):
        return self.values.equals(other.values)


def factor_product(factor1, factor2):
    new_variables = set(factor1.variables).union(set(factor2.variables))
    joined_variables = set.intersection(set(factor1.variables), set(factor2.variables))
    tmp1 = factor1.values.reset_index()
    tmp2 = factor2.values.reset_index()
    tmp = pd.merge(tmp1, tmp2, on=sorted(joined_variables))
    tmp['phi'] = np.multiply(tmp['phi_x'], tmp['phi_y'])
    del tmp['phi_x']
    del tmp['phi_y']
    tmp = tmp.set_index(sorted(new_variables)).sort_index()
    return Factor(tmp)


def factor_marginalization(factor, variables):
    kept_variables = list(set(factor.variables) - set(variables))
    tmp = factor.values.reset_index()
    df = tmp.groupby(by=kept_variables).aggregate(sum)
    for v in variables:
        del df[v]
    return Factor(df)


def observe_evidence(factor, evidence):
    tmp = factor.values.reset_index()
    for variable, value in evidence.items():
        if variable in tmp.columns:
            tmp.loc[tmp[variable] != value, 'phi'] = 0.
    return Factor(tmp.set_index(factor.variables))


def compute_joint_distribution(factors):
    if len(factors) == 1:
        return factors[0]
    return factors[0] * compute_joint_distribution(factors[1:])
