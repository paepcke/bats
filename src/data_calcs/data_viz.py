'''
Created on Apr 28, 2024

@author: paepcke
'''
from data_calcs.data_prep import DataPrep
from data_calcs.universal_fd import UniversalFd
import os

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.cluster.tests.test_affinity_propagation import n_clusters

class DataViz:
    '''
    Visualizations such as Tsne plots.
    '''

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, infile):
        '''
        Infile must be either a string path, or a 
        file-like object. This file will be the subject
        of all visualizations performed by this instance.
        '''
        
        
        if type(infile) != str:
            self.infile_path = infile.name
        else:
            self.infile_path = infile
        
        if not os.path.exists(infile):
            raise FileNotFoundError(f"File '{self.infile_path}' not reachable (maybe forgot to mount remote machine?)")
            
        self.infile = infile
        
        
    #------------------------------------
    # run_tsne
    #-------------------
    
    def run_tsne(self, 
                 num_points=10000, 
                 num_dims=50,
                 point_id_col='file_id',
                 perplexity=None,
                 sort_by_bat_variance=True
                 ):
        '''
        Infile must be a .feather, .csv. or .csv.gz file
        of SonoBat measures. The file is loaded into a DataFrame. 
        
        This function pulls the first num_dims columns from that
        df, and limits the T-sne embedding input to those columns.
        If num_dims is None, all input columns are retained. 
        
        If sort_by_bat_variance is True, columns must be SonoBat
        program bat chirp measurement names. With that switch 
        being True, the columns of the input df are sorted by 
        decreasing variance over the Jasper Ridge bat recordings.
        The num_dims columns are then taken from that sorted list
        of measurement result columns. 
        
        after sorting the columns by decreasing variance.
        That is the num_dims'th ranked variance measures are used
        for each chirp. 
        
        Uses num_points (i.e. num_points rows) from the df.
        
        Tsne is run over the resulting num_points x num_dims dataframe.  
        
        :param num_points: number of chirps to include from the 
            given self.infile.
        :type num_points: int
        :param num_dims: number of measures of each chirp to use.
            I.e. how many columns.
        :type num_dims:
        :param sort_by_bat_variance: if True, all column names must
            be SonoBat measure names. The num_dims number of columns
            will be selected from the input dataframe such that they
            have highest rank in variance over the bat recordings.
        :type sort_by_bat_variance: bool
        :result the T-sne embeddings
        :rtype pd.DataFrame
        '''
              
        # Sort by variance, and cut off below-threshold
        # columns:
        if sort_by_bat_variance:
            df_all_rows = DataPrep.measures_by_var_rank(self.infile, min_var_rank=num_dims)
        else:
            fd = UniversalFd(self.infile, 'r')
            df_all_rows = fd.asdf()
        
        # Keep only the wanted columns:
        if type(num_points) == int: 
            df = df_all_rows.iloc[0:num_points]
        elif num_points is None:
            df = df_all_rows
        else:
            raise TypeError(f"The num_points arg must be None or an integer, not {num_points}")

        # Perplexity must be less than number of points:
        if perplexity is not None:
            if type(perplexity) != float:
                raise TypeError(f"Perplexity must be None, or float, not {perplexity}")
            if perplexity >= len(df):
                perplexity = float(len(df) - 1)
        else:
            # Mirror the TSNE constructor's default:
            perplexity = min(30.0, len(df)-1)
        
        tsne_obj = TSNE(n_components=2, init='pca', perplexity=perplexity)
        embedding_arr = tsne_obj.fit_transform(df)
        
        # For each embedding point, add the point identifier,
        # which is a column from the original df, i.e. 
        # which is the value in the given point_id_col:
        tsne_df = pd.DataFrame(embedding_arr, index=df[point_id_col], columns=['tsne_x', 'tsne_y'])
        return tsne_df
    
    #------------------------------------
    # cluster_tsne
    #-------------------
    
    def cluster_tsne(self, tsne_df, n_clusters=None, cluster_range=range(2,10)):
        '''
        Computes Kmeans on the tsne dataframe. The returned
        Kmeans object can be interrogated for the assignment
        of each Tsne-embedded data point to a cluster. Example
        for a Tsne df of 10 rows (i.e. 10 datapoints in both the
        orginal df, and the Tsne embedding):
        
            kmeans-obj.labels_ => expected = [0, 0, 2, 0, 2, 1, 0, 2, 2, 1]
            
        The numbers are cluster labels.
        
        Cluster centers in Tsne space are available as:
        
            kmeans-obj.cluster_centers():
          returns like: array([[ 17.117634 ,  23.18263  ],
                               [  6.5119925, -38.010742 ],
                               [-52.590286 ,   3.915401 ]], dtype=float32)
      
        If n_clusters is None, this method finds the best n_cluster by
        the silhouette method. 
        
        In this case the silhouette coefficients for each tested n_clusters 
        will be available in the 
        
                self.silhouettes
                
        attribute. If an n_cluster is provided by the caller, this attribute
        will be undefined.
        
        Also in the case of the given n_clusters being None, the computed 
        silhouette coefficient for each tried n_clusters will be available
        in the list attribute:
        
                self.n_cluster_range
        
        :param tsne_df: dataframe of points in Tsne space, as returned
            by the run_tsne() method.
        :type tsne_df: pd.DataFrame
        :param n_clusters: number of clusters to find. If None,
            a silhouette analysis is performed over n_cluster in [2...10].
            The n_cluster resulting in the largest average silhouette coefficient
            is chosen as n_clusters 
        :type n_clusters: UNION[None | int]
        :param cluster_range: Used when n_clusters is None. The range of
            n_cluster values to test for leading to optimal clustering.
        :type cluster_range: range
        :return: a KMeans instance
        :rtype sklearn.cluster.KMeans
        '''
        self.n_cluster_range = list(cluster_range)
        
        np_arr = tsne_df.to_numpy()
        if n_clusters is not None:
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(np_arr)
            return kmeans

        # Need to find best number of clusters via
        # silhouette method:
                
        # Dict attemt_number : kmeans_result
        attempts = {}
        # List of silhouette coefficients resulting
        # from each n_cluster: 
        self.silhouettes = []
        
        for i, n_clusters in enumerate(cluster_range):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(np_arr)
            # Save the computed kmeans as the i'th iteration:
            attempts[i] = kmeans
            # Compute the silhouette coefficient for this kmeans,
            # and append it to the silhouttes list in the i'th position: 
            self.silhouettes.append(silhouette_score(np_arr, kmeans.labels_))
        
        # Find the index of the largest (average) silhouette:
        max_silhouette_idx = self.silhouettes.index(max(self.silhouettes))
        # Now we know which of the kmeans results to use:
        kmeans = attempts[max_silhouette_idx]
        return kmeans
        
    #------------------------------------
    # plot_tsne
    #-------------------
    
    def plot_tsne(self, tsne_df):
        
        fig, ax = plt.subplots()
        cols = tsne_df.columns
        ax.scatter(tsne_df[cols[0]], tsne_df[cols[1]])
        # If you want x-labels at 45deg:
        # ax.set_xticklabels(ax.get_xticks(), rotation = 45)
        fig.show()
        
            
    