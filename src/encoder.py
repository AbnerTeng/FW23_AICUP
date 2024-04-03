"""
Beta Target Encoder

Returns:
    _type_: _description_
"""
import numpy as np
import pandas as pd


class BetaEncoder:
    """
    Beta encoder for categorical features
    """
    def __init__(self, group) -> None:
        self.group = group
        self.stats = None
        self.prior_mean = None


    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """fitting the encoder

        Args:
            df (pd.DataFrame)
            target_col (str)
        """
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.reset_index(level=0, inplace=True)
        self.stats = stats


    def transform(
            self, df: pd.DataFrame, stat_type: str, n_min: int = 10
        ) -> pd.Series:
        """
        transform the encoder

        Args:
            df (pd.DataFrame): _description_
            stat_type (str): _description_
            n_min (int, optional): _description_. Defaults to 1.

        Returns:
            pd.Series: _description_
        """
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['sum'].copy()
        big_n = df_stats['count'].copy()
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        big_n[nan_indexs] = 1.0

        big_n_prior = np.maximum(n_min - big_n, 0)
        alpha_prior = self.prior_mean * big_n_prior
        beta_prior  = (1 - self.prior_mean) * big_n_prior

        alpha =  alpha_prior + n
        beta =  beta_prior + big_n - n

        stats_dict = {
            "mean": {
                'nom': alpha,
                'denom': alpha + beta
            },
            "mode": {
                'nom': alpha - 1,
                'denom': alpha + beta - 2
            },
            "median": {
                'nom': alpha - 1/3,
                'denom': alpha + beta - 2/3
            },
            "var": {
                'nom': alpha * beta,
                'denom': (alpha + beta) ** 2 * (alpha + beta + 1)
            },
            "skewness": {
                'nom': 2 * (beta - alpha) * np.sqrt(alpha + beta + 1),
                'denom': (alpha + beta + 2) * np.sqrt(alpha * beta)
            },
            "kurtosis": {
                'nom': 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2),
                'denom': alpha*beta*(alpha+beta+2)*(alpha+beta+3)
            }
        }

        value = stats_dict[stat_type]['nom'] / stats_dict[stat_type]['denom']
        value[np.isnan(value)] = np.nanmedian(value)
        return value
