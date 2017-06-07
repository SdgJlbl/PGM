#! /usr/bin/env python
import operator
from functools import reduce
import numpy as np
import pandas as pd


class Factor(object):
    """
    A class representing a factor using a pandas DataFrame.

    A factor defined on a set of discrete random variables is a function associating a value in R for each possible
    combination of values of the variables.

    These values are stored in a column named 'phi'. The variables names are the level names of the row multiindex.
    The variables values are 0, 1, ..., n where n is the cardinality of the variable.

    Attributes
    ----------
    values : pandas.DataFrame
        A dataframe containing the values of the factor; variables are in the lexicographic order of the names; rows are
        ordered according to the variable values.
        Eg :
        -----------------
        |         | phi |
        -----------------
        | v1 | v2 |     |
        -----------------
        |  0 |  0 | 0.1 |
        -----------------
        |  0 |  1 | 0.9 |
        -----------------
        |  1 |  0 | 0.7 |
        -----------------
        |  1 |  1 | 0.3 |
        -----------------

    """
    def __init__(self, df):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe containing the values of the factor. The dataframe must respect the above-mentioned convention
            (one column named 'phi', variables names are the multiindex level names, variables values are
            range(0, variable_cardinality))
        """
        assert df.columns == ['phi']
        names = df.index.names
        self.values = df.reset_index().set_index(sorted(names)).sort_index()

    @classmethod
    def from_scratch(cls, variables, variable_cardinalities, values=None):
        """ Construct a factor from a list of variables with their associated cardinalities and a list of values.
        The factor values are ordered so that they correspond to the variables values in the tuple ordering.

        Eg : values[0] = f((0, 0, ..., 0)), values[1] = f((0, 0, ..., 0, 1))

        Parameters
        ----------
        variables : list of str
            List of the variables names
        variable_cardinalities : list of int
            List of the cardinality of each variable, ordered as given in 'variables' list. Values of the variables are
             range(0, variables_cardinalities[i])
        values : list of float
            Values taken by the factor on each combination of variables values

        Returns
        -------
        Factor

        """
        if not values:
            values = np.ones(np.prod(np.array(variable_cardinalities)))
        else:
            assert len(values) == reduce(operator.mul, variable_cardinalities)
        index = pd.MultiIndex.from_product([range(c) for c in variable_cardinalities], names=variables)
        df = pd.DataFrame(data=values, index=index, columns=['phi'])
        return cls(df)

    @property
    def variables(self):
        """ list of str : list of the variables names"""
        return self.values.index.names

    @property
    def variable_cardinalities(self):
        """ list of int : list of the variables cardinalities """
        return list(map(len, self.values.index.levels))

    def __mul__(self, other):
        return factor_product(self, other)

    def __eq__(self, other):
        return self.values.equals(other.values)


def factor_product(factor1, factor2):
    """ Compute the product of two factors

    Parameters
    ----------
    factor1, factor2 : Factor

    Returns
    -------
    Factor
        The resulting product

    """
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
    """ Marginalize a vector over the given variables

    Parameters
    ----------
    factor : Factor
        The current factor
    variables : list of str
        List of variable names

    Returns
    -------
    Factor
        A new factor in which given variables have been marginalised

    """
    kept_variables = list(set(factor.variables) - set(variables))
    tmp = factor.values.reset_index()
    df = tmp.groupby(by=kept_variables).aggregate(sum)
    for v in variables:
        del df[v]
    return Factor(df)


def observe_evidence(factor, evidence):
    """ Modify the values in a factor to reflect observed evidence

    The values which are incompatible with the observed evidence are set to 0. The returned factor is NOT normalized.

    Parameters
    ----------
    factor : Factor
        The current factor
    evidence : dict
        A dictionary of variables names / observed values

    Returns
    -------
    Factor
        A copy of the input factor in which appropriate values are modified to reflect evidence
    """
    tmp = factor.values.reset_index()
    for variable, value in evidence.items():
        if variable in tmp.columns:
            tmp.loc[tmp[variable] != value, 'phi'] = 0.
    return Factor(tmp.set_index(factor.variables))


def compute_joint_distribution(factors):
    """ Compute the joint distribution for a sequence of factors, using the chain rule

    This correspond to a joint distribution represented by a Bayesian network such as :
        factors[0] -> factors[1] -> factors[2]

    Parameters
    ----------
    factors : list
        List of Factors representing valid conditional probabilities distributions

    Returns
    -------
    Factor
        The resulting joint distribution for the chain

    """
    if len(factors) == 1:
        return factors[0]
    return factors[0] * compute_joint_distribution(factors[1:])
