import pandas as pd
import numpy as np
from fractions import Fraction


class DataAnalysisBackend:
    """
    Backend class for statistical data analysis operations.
    Handles all data processing, encoding, and statistical computations.
    """
    
    def __init__(self, data):
        """
        Initialize the backend with data
        
        Parameters:
        -----------
        data : dict or DataFrame
            The data to analyze
        """
        self.df = pd.DataFrame(data)
        self.data = data
    
    def encode_binary_response(self, column, positive_value, negative_value):
        """
        Encode a binary response column
        
        Parameters:
        -----------
        column : str
            Column name to encode
        positive_value : str
            Value to encode as '10'
        negative_value : str
            Value to encode as '01'
            
        Returns:
        --------
        pandas.Series : Encoded column
        """
        return self.df[column].apply(
            lambda x: '10' if x == positive_value else '01'
        )
    
    def encode_ordinal_response(self, column, ordered_categories):
        """
        Encode an ordinal response column using binary expansion
        
        Parameters:
        -----------
        column : str
            Column name to encode
        ordered_categories : list
            Ordered list of categories (lowest to highest)
            
        Returns:
        --------
        dict : Mapping of categories to binary codes
        pandas.Series : Encoded column
        """
        n = len(ordered_categories)
        binary_codes = [
            "".join("1" if j <= i else "0" for j in range(n)) 
            for i in range(n)
        ]
        freq_binary = dict(zip(ordered_categories, binary_codes))
        encoded = self.df[column].map(freq_binary)
        return freq_binary, encoded
    
    def get_disjunctive_coding(self, exclude_columns=None):
        """
        Get complete disjunctive coding (dummy variables)
        
        Parameters:
        -----------
        exclude_columns : list, optional
            Columns to exclude from coding
            
        Returns:
        --------
        pandas.DataFrame : Disjunctive coded matrix
        """
        if exclude_columns is None:
            exclude_columns = []
        df_subset = self.df.drop(columns=exclude_columns)
        return pd.get_dummies(df_subset).astype(int)
    
    def get_ordinal_disjunctive_coding(self, response1_col, response2_col, 
                                       ordinal_mapping, chercheur_col='Chercheur'):
        """
        Get disjunctive coding with ordinal expansion for specified column
        
        Parameters:
        -----------
        response1_col : str
            Binary response column name
        response2_col : str
            Ordinal response column name
        ordinal_mapping : dict
            Mapping of categories to binary codes
        chercheur_col : str
            Column to use as index
            
        Returns:
        --------
        pandas.DataFrame : Ordinal disjunctive coded matrix
        """
        # Standard dummies for response 1
        reponse1_dummies = pd.get_dummies(
            self.df[response1_col], prefix=response1_col
        ).astype(int)
        
        # Ordinal expansion for response 2
        reponse2_ordinal = pd.DataFrame(dtype=int)
        for idx, value in enumerate(self.df[response2_col].tolist()):
            binary_code = ordinal_mapping[value]
            for bit_pos, bit in enumerate(binary_code):
                col_name = f"{response2_col}_{bit_pos+1}"
                reponse2_ordinal.loc[idx, col_name] = int(bit)
        
        reponse2_ordinal = reponse2_ordinal.fillna(0).astype(int)
        
        # Concatenate
        X_ordinal = pd.concat([
            reponse1_dummies.reset_index(drop=True), 
            reponse2_ordinal.reset_index(drop=True)
        ], axis=1).astype(int)
        
        X_ordinal.index = self.df[chercheur_col].tolist()
        return X_ordinal
    
    def compute_burt_matrix(self, X):
        """
        Compute Burt matrix (X'X)
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Disjunctive coded matrix
            
        Returns:
        --------
        pandas.DataFrame : Burt matrix
        """
        X_matrix = X.to_numpy()
        Burt = X_matrix.T @ X_matrix
        return pd.DataFrame(Burt, index=X.columns, columns=X.columns)
    
    def compute_distance_matrix(self, X_mat, individuals):
        """
        Compute distance matrix: d(I,J) = (b+c)/(a+b+c+d)
        
        Parameters:
        -----------
        X_mat : numpy.ndarray
            Binary matrix
        individuals : list
            Individual labels
            
        Returns:
        --------
        tuple : (distance_df, fraction_df)
        """
        n_ind = X_mat.shape[0]
        dist = np.zeros((n_ind, n_ind))
        frac_tab = [[None]*n_ind for _ in range(n_ind)]
        
        for i in range(n_ind):
            for j in range(n_ind):
                if i == j:
                    dist[i, j] = 0.0
                    frac_tab[i][j] = "0"
                else:
                    a = int(np.sum((X_mat[i] == 1) & (X_mat[j] == 1)))
                    b = int(np.sum((X_mat[i] == 1) & (X_mat[j] == 0)))
                    c = int(np.sum((X_mat[i] == 0) & (X_mat[j] == 1)))
                    d = int(np.sum((X_mat[i] == 0) & (X_mat[j] == 0)))
                    num = b + c
                    den = a + b + c + d
                    if den > 0:
                        dist[i, j] = num / den
                        frac_tab[i][j] = str(Fraction(num, den))
                    else:
                        dist[i, j] = 0.0
                        frac_tab[i][j] = "0"
        
        dist_df = pd.DataFrame(dist, index=individuals, columns=individuals)
        frac_df = pd.DataFrame(frac_tab, index=individuals, columns=individuals)
        return dist_df, frac_df
    
    
    def compute_dissimilarity_matrix(self, X_mat, individuals):
        """
        Compute dissimilarity matrix: s(I,J) = (a+d)/(a+b+c+d)
        
        Parameters:
        -----------
        X_mat : numpy.ndarray
            Binary matrix
        individuals : list
            Individual labels
            
        Returns:
        --------
        tuple : (dissimilarity_df, fraction_df)
        """
        n_ind = X_mat.shape[0]
        sim = np.zeros((n_ind, n_ind))
        frac_tab = [[None]*n_ind for _ in range(n_ind)]
        
        for i in range(n_ind):
            for j in range(n_ind):
                if i == j:
                    sim[i, j] = 1.0
                    frac_tab[i][j] = "1"
                else:
                    a = int(np.sum((X_mat[i] == 1) & (X_mat[j] == 1)))
                    b = int(np.sum((X_mat[i] == 1) & (X_mat[j] == 0)))
                    c = int(np.sum((X_mat[i] == 0) & (X_mat[j] == 1)))
                    d = int(np.sum((X_mat[i] == 0) & (X_mat[j] == 0)))
                    num = a + d
                    den = a + b + c + d
                    if den > 0:
                        sim[i, j] = num / den
                        frac_tab[i][j] = str(Fraction(num, den))
                    else:
                        sim[i, j] = 0.0
                        frac_tab[i][j] = "0"
        
        sim_df = pd.DataFrame(sim, index=individuals, columns=individuals)
        frac_df = pd.DataFrame(frac_tab, index=individuals, columns=individuals)
        return sim_df, frac_df
    
    def compute_contingency_table(self, col1, col2):
        """
        Compute contingency table between two columns
        
        Parameters:
        -----------
        col1 : str
            First column name
        col2 : str
            Second column name
            
        Returns:
        --------
        pandas.DataFrame : Contingency table
        """
        return pd.crosstab(self.df[col1], self.df[col2])
    
    def compute_row_profiles(self, contingency):
        """
        Compute row profiles with fractions
        
        Parameters:
        -----------
        contingency : pandas.DataFrame
            Contingency table
            
        Returns:
        --------
        tuple : (row_profiles_df, cloud_display_df)
        """
        row_totals = contingency.sum(axis=1)
        n_total = contingency.values.sum()
        row_profiles = contingency.div(row_totals, axis=0)
        
        # Build display with fractions
        cloud_display = pd.DataFrame(
            index=row_profiles.index, 
            columns=row_profiles.columns
        )
        for i in row_profiles.index:
            for j in row_profiles.columns:
                numerator = int(contingency.loc[i, j])
                denom = int(row_totals.loc[i])
                cloud_display.loc[i, j] = str(Fraction(numerator, denom)) if denom > 0 else "0"
        
        # Add f_i column
        cloud_display['f_i'] = [
            str(Fraction(int(row_totals.loc[r]), int(n_total))) 
            for r in cloud_display.index
        ]
        
        return row_profiles, cloud_display
    
    def compute_column_profiles(self, contingency):
        """
        Compute column profiles with fractions
        
        Parameters:
        -----------
        contingency : pandas.DataFrame
            Contingency table
            
        Returns:
        --------
        tuple : (col_profiles_df, col_cloud_display_df)
        """
        col_totals = contingency.sum(axis=0)
        n_total = contingency.values.sum()
        col_profiles = contingency.div(col_totals, axis=1)
        
        # Build display with fractions
        col_cloud_display = pd.DataFrame(
            index=col_profiles.index, 
            columns=col_profiles.columns
        )
        for i in col_profiles.index:
            for j in col_profiles.columns:
                numerator = int(contingency.loc[i, j])
                denom = int(col_totals.loc[j])
                col_cloud_display.loc[i, j] = str(Fraction(numerator, denom)) if denom > 0 else "0"
        
        # Add f_j row
        f_j_row = [
            str(Fraction(int(col_totals.loc[c]), int(n_total))) 
            for c in col_profiles.columns
        ]
        col_cloud_display.loc['f_j'] = f_j_row
        
        return col_profiles, col_cloud_display
    
    def compute_chi2_distance_rows(self, contingency):
        """
        Compute χ² distance between row profiles
        
        Parameters:
        -----------
        contingency : pandas.DataFrame
            Contingency table
            
        Returns:
        --------
        pandas.DataFrame : χ² distance matrix for rows
        """
        total = contingency.values.sum()
        freq = contingency / total
        row_masses = freq.sum(axis=1)
        col_masses = freq.sum(axis=0)
        row_profiles = freq.div(row_masses, axis=0).fillna(0)
        
        n = row_profiles.shape[0]
        d2 = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                diff = row_profiles.iloc[i] - row_profiles.iloc[k]
                d2[i, k] = np.sum((diff**2) / col_masses)
        
        return pd.DataFrame(
            d2, 
            index=row_profiles.index, 
            columns=row_profiles.index
        )
    
    def compute_chi2_distance_cols(self, contingency):
        """
        Compute χ² distance between column profiles
        
        Parameters:
        -----------
        contingency : pandas.DataFrame
            Contingency table
            
        Returns:
        --------
        pandas.DataFrame : χ² distance matrix for columns
        """
        total = contingency.values.sum()
        freq = contingency / total
        row_masses = freq.sum(axis=1)
        col_masses = freq.sum(axis=0)
        col_profiles = freq.div(col_masses, axis=1).fillna(0)
        
        n = col_profiles.shape[1]
        d2 = np.zeros((n, n))
        for j in range(n):
            for l in range(n):
                diff = col_profiles.iloc[:, j] - col_profiles.iloc[:, l]
                d2[j, l] = np.sum((diff**2) / row_masses)
        
        return pd.DataFrame(
            d2, 
            index=col_profiles.columns, 
            columns=col_profiles.columns
        )