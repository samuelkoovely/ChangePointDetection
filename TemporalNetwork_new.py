"""
#
# flow stability
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""

import os
import pandas as pd
import numpy as np
from scipy.sparse import (lil_matrix, dok_matrix, diags, eye, isspmatrix_csr, isspmatrix,
                          csr_matrix, coo_matrix, csc_matrix)
from scipy.sparse.linalg import expm, eigsh
from scipy.linalg import expm as d_expm
from scipy.linalg import logm as d_logm
from scipy.sparse.csgraph import connected_components
import gzip
from SparseStochMat import sparse_stoch_mat, inplace_csr_row_normalize

from parallel_expm import compute_subspace_expm_parallel

from functools import partial

import time
import pickle
from math import exp

class ContTempNetwork(object):
    """ Continuous time temporal network
    
    Parameters:
    -----------
    
    source_nodes: Python list
        List of source nodes of each event, ordered according to `starting_times`.
    
    target_nodes: Python list
        List of target nodes of each event.
        
    starting_times: Python list
        List of starting times of each event.
        
    ending_times: Python list
        List of ending times of each event.
        
    extra_attrs: Dict
        Extra event attributes. Must be given in a dict with {attr_name: list_of_values},
        where list_of_values has the same order and length as `source_nodes`.
        
    relabel_nodes: boolean
        Relabel nodes from 0 to num_nodes and save original labels in 
        self.node_to_label_dict. Default is `True`
        
    reset_event_table_index: boolean
        Reset the index of the `events_table` DataFrame. Default is `True`.
        
    node_to_label_dict: Python dict
        If `relabel_nodes` is `False, this can be used to save the original labels
        of the nodes.
        
    merge_overlapping_events: boolean
        Check for overlapping events (between the same pair of nodes) 
        and merges them. Default is `False`.
        
    events_table: Pandas Dataframe
        Dataframe with columns 'source_nodes', 'target_nodes', 'starting_times'
        and 'ending_times' and index corresponding to event index. Used for
        instantiating a new ConTempNetwork from the event_table of an other one.
    
    
    """
    def __init__(self, source_nodes=[],
                       target_nodes=[],
                       starting_times=[],
                       ending_times=[],
                       extra_attrs=None,
                       relabel_nodes=True,
                       reset_event_table_index=True,
                       node_to_label_dict=None,
                       merge_overlapping_events=False,
                       events_table=None):
    
        if events_table is None:
            assert len(source_nodes) == len(target_nodes) == \
                   len(starting_times) == len(ending_times)
                    
            if relabel_nodes:
               # relabel nodes from 0 to num_nodes and save
                # original labels in self.node_to_label_dict
                all_nodes = set()
                all_nodes.update(source_nodes)
                all_nodes.update(target_nodes)
                self.label_to_node_dict = {l : n for n, l in enumerate(sorted(all_nodes))}
                self.node_to_label_dict = {n : l for l, n in self.label_to_node_dict.items()}
                
                source_nodes = [self.label_to_node_dict[n] for n in source_nodes]
                target_nodes = [self.label_to_node_dict[n] for n in target_nodes]
            else:
                self.node_to_label_dict=node_to_label_dict
                
            data={'source_nodes' : source_nodes,
                                                  'target_nodes' : target_nodes,
                                                  'starting_times' : starting_times,
                                                  'ending_times' : ending_times}
            columns=['source_nodes','target_nodes',
                                             'starting_times','ending_times']
            
            if extra_attrs is not None:
                assert isinstance(extra_attrs, dict)
                
                for attr_name, val_list in extra_attrs.items():
                    assert len(val_list) == len(source_nodes)
                    data[attr_name] = val_list
                    columns.append(attr_name)
            
            self.events_table = pd.DataFrame(data=data,
                                             columns=columns)
            
            self.events_table.sort_values(by=['starting_times', 'ending_times'],
                                          inplace=True)
            
        else:
            self.events_table = events_table
            reset_event_table_index = False
            if relabel_nodes:
                all_nodes = set()
                all_nodes.update(events_table.source_nodes.tolist())
                all_nodes.update(events_table.target_nodes.tolist())
                self.label_to_node_dict = {l : n for n, l in enumerate(sorted(all_nodes))}
                self.node_to_label_dict = {n : l for l, n in self.label_to_node_dict.items()}
            else:
                self.node_to_label_dict=node_to_label_dict

                        
        if reset_event_table_index:
            self.events_table.reset_index(inplace=True, drop=True)

        self.node_array = np.sort(pd.unique(self.events_table[['source_nodes',
                                    'target_nodes']].values.ravel('K')))
        
        self.num_nodes = self.node_array.shape[0]
        
        self.num_events = self.events_table.shape[0]
        
        self.start_time = self.events_table.starting_times.min()
    
        self.end_time = self.events_table.ending_times.max()
        
        self.events_table['durations'] = self.events_table.ending_times - \
                            self.events_table.starting_times
        
        # to record compute times
        self._compute_times = {}
        
        self._overlapping_events_merged = False
        if merge_overlapping_events:
            num_merged = 1
            while num_merged != 0:
                num_merged = self._merge_overlapping_events()
            self._overlapping_events_merged = True
            
        
        self.is_directed = False
        self.instantaneous_events = False


    def __repr__(self):
        return str(self.__class__) + \
              f' with {self.num_nodes} nodes and {self.num_events} events'
        
    def save(self, filename, 
             matrices_list=None,
             attributes_list=None):
        """ save graph event_table and matrices as pickle file
        
        Parameters:
        -----------
        
        filename: str
            Filename where to save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to save.
            The default list is:
                `matrices_list = ['laplacians',
                                    'adjacencies',
                                    'inter_T',
                                    'inter_T_lin',
                                    'T',
                                    'T_lin',
                                    '_stationary_trans',
                                    'delta_inter_T',
                                    'delta_inter_T_lin',]`
        attributes_list: list of strings
            List of attribute names to save.
            The default list is:
                `attributes_list = ['node_to_label_dict',
                                  'events_table',
                                  'times',
                                  'time_grid',
                                  'num_nodes',
                                  '_compute_times',
                                  '_t_start_laplacians',
                                  '_k_start_laplacians',
                                  '_t_stop_laplacians',
                                  '_k_stop_laplacians',
                                  '_overlapping_events_merged',]`
                                    
            
        """
        
        
        save_dict = dict()
        
        
        matrices = ['laplacians',
                      'adjacencies',
                      'inter_T',
                      'inter_T_lin',
                      'T',
                      'T_lin',
                      '_stationary_trans',
                      'delta_inter_T',
                      'delta_inter_T_lin',]
        
        if matrices_list is None:
            matrices_list = matrices
        
        attributes = ['node_to_label_dict',
                      'events_table',
                      'times',
                      'time_grid',
                      'num_nodes',
                      '_compute_times',
                      '_t_start_laplacians',
                      '_k_start_laplacians',
                      '_t_stop_laplacians',
                      '_k_stop_laplacians',
                      '_overlapping_events_merged',
                      'is_directed']
        
        if attributes_list is None:
            attributes_list = attributes
        
        for attr in attributes_list:
            if hasattr(self, attr):
                save_dict[attr] = getattr(self,attr)
                

        for mat in matrices_list:
            if hasattr(self, mat):
                save_dict[mat] = getattr(self,mat)
                

        with open(os.path.splitext(filename)[0] + '.pickle', 'wb') as fopen:
            pickle.dump(save_dict, fopen)
                        
        
    @classmethod 
    def load(cls, filename,
             matrices_list=None,
             attributes_list=None):
        """ load network from file and returns a ContTempNetwork instance.
        
        Parameters:
        -----------
        
        filename: str
            Filename where the network is saved save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to load.
            The default list is:
                `matrices_list = ['laplacians',
                                    'adjacencies',
                                    'inter_T',
                                    'inter_T_lin',
                                    'T',
                                    'T_lin',
                                    '_stationary_trans',
                                    'delta_inter_T',
                                    'delta_inter_T_lin',]`
        attributes_list: list of strings
            List of attribute names to load.
            The default list is:
                `attributes_list = ['node_to_label_dict',
                                  'events_table',
                                  'times',
                                  'time_grid',
                                  'num_nodes',
                                  '_compute_times',
                                  '_t_start_laplacians',
                                  '_k_start_laplacians',
                                  '_t_stop_laplacians',
                                  '_k_stop_laplacians',
                                  '_overlapping_events_merged',]`
                                    
            
            
        """
        
        matrices = ['laplacians',
                      'adjacencies',
                      'inter_T',
                      'inter_T_lin',
                      'T',
                      'T_lin',
                      '_stationary_trans',
                      'delta_inter_T',
                      'delta_inter_T_lin',]
        
        if matrices_list is None:
            matrices_list = matrices
        
        attributes = ['node_to_label_dict',
                      'events_table',
                      'times',
                      'time_grid',
                      'num_nodes',
                      '_compute_times',
                      '_t_start_laplacians',
                      '_k_start_laplacians',
                      '_t_stop_laplacians',
                      '_k_stop_laplacians',
                      '_overlapping_events_merged',
                      'is_directed'] 
                    
        if attributes_list is None:
            attributes_list = attributes
            
            
        # all in a pickle file

        graph_dict = pd.read_pickle(os.path.splitext(filename)[0] +'.pickle')
        
        events_table = graph_dict.pop('events_table')

        net = cls(events_table=events_table,
                           relabel_nodes=False,
                           node_to_label_dict=graph_dict.pop('node_to_label_dict'))

        for k,val in graph_dict.items():
            if k in matrices_list:
                setattr(net, k, val)
            if k in attributes_list:
                setattr(net, k, val)
            
        return net
        
        
    def save_inter_T(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                                   compressed=False,
                                   save_delta=False,
                                   replace_existing=False):
        """ 
        
            
            Saves all the inter transition matrices in `self.inter_T[lamda]` 
            in a pickle file togheter  with a dictionary including parameters: 
            `_k_start_laplacians`, `_k_stop_laplacians`, `_t_start_laplacians`,
            `_t_stop_laplacians`, `_t_stop_laplacians`, `times_k_start_to_k_stop+1` 
            (= self.times.values[self._k_start_laplacians:self._k_stop_laplacians+1])
            `num_nodes` and `_compute_times`.
            
            if `save_delta` is True, creates delta_inter_T if it is 
            not already present and saves it together with 
            inter_T[lamda][0] in a pickle file.
            otherwise, saves inter_T[lamda] directly (good if used with
            sparse_stoch_matrices).
            
            Parameters:
            -----------
            
            filename: str
                Filename where the data is saved (.pickle or .gz).
            lamda: float or int, optional.
                Use to save to results for a specific lamba. If None, the results
                for all the computed lambdas will be saved. Default is None.
            round_zeros: bool.
                If True, values smaller than tol*max(abs(inter_T_k)) will be set to zero
                to preserve sparsity. Default is True.
            tol: float
                See round_zeros. Default is 1e-8.
            compressed: bool
                Used to compress the file with gzip. Default is False.
            save_delta: bool
                If True, creates delta_inter_T if it is 
                not already present and saves it together with 
                inter_T[lamda][0]. Only the differences between two consecutives
                inter_Ts are saved in order to minimize file size. 
                Must not be used if `use_sparse_stoch` was used in `compute_inter_transition_matrices`.
            replace_existing: bool
                If True, erase and replace files if they already exists. 
                Default is False.
            
            Returns:
            --------
            
            None
            
                        
        """
        assert hasattr(self, 'inter_T'), f'PID {os.getpid()} : nothing saved, compute inter_T first.'
        
        
        ext = os.path.splitext(filename)[-1]

        file = filename
            
        if compressed:                            
            if ext != '.gz':
                file += '.gz'
        else:
            if ext != '.pickle':
                file += '.pickle'
                
        skipping = False
        if os.path.exists(file):
            if replace_existing:
                print('PID ', os.getpid(), f' : , file {file} already exists, replacing it.')
            else:
                print('PID ', os.getpid(), f' : , file {file} already exists, skipping.')
                skipping = True
            
        if not skipping:
            
        
            save_dict = {}
            save_dict['_k_start_laplacians'] = self._k_start_laplacians
            save_dict['_k_stop_laplacians'] = self._k_stop_laplacians
            save_dict['_t_start_laplacians'] = self._t_start_laplacians
            save_dict['_t_stop_laplacians'] = self._t_stop_laplacians
            save_dict['times_k_start_to_k_stop+1'] = self.times.values[\
                                self._k_start_laplacians:self._k_stop_laplacians+1]
            save_dict['num_nodes'] = self.num_nodes
            save_dict['_compute_times'] = self._compute_times
            
            if save_delta:
                assert not isinstance(self.inter_T[lamda][0], 
                                  sparse_stoch_mat),\
                                "inter_T must not be sparse_stoch_mat"
                
                if not hasattr(self, 'delta_inter_T') and \
                                    hasattr(self, 'inter_T'):
                    if lamda is not None:
                        self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)
                    else:
                        #computes it for all lamda
                        for lamda in self.inter_T.keys():
                            self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)
                            
                if (lamda is not None) and (lamda not in self.delta_inter_T.keys()):
                    self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)
                
                if hasattr(self, 'delta_inter_T'):
                    save_dict['inter_T'] = dict()
                    save_dict['is_delta_trans'] = True
                    
                    if lamda is None:
                        lamdas = self.delta_inter_T.keys()
                    else:
                        lamdas = [lamda]
                    
                    for lamda in lamdas:
                        save_dict['inter_T'][lamda] = dict()
                        save_dict['inter_T'][lamda]['delta_inter_T'] = self.delta_inter_T[lamda]
                        save_dict['inter_T'][lamda]['trans_mat0'] = self.inter_T[lamda][0].copy()
                        if round_zeros:
                            set_to_zeroes(save_dict['inter_T'][lamda]['trans_mat0'],
                                          tol=tol)
                
                text = "delta trans mats"
                        
            else:
                
                save_dict['inter_T'] = dict()
                save_dict['is_sparse_stoch'] = True
                
                if lamda is None:
                    lamdas = self.inter_T.keys()
                else:
                    lamdas = [lamda]
                
                for lamda in lamdas:
                    assert isinstance(self.inter_T[lamda][0], 
                                      sparse_stoch_mat),\
                                    "inter_T needs to be sparse_stoch_mat"
                    
                    save_dict['inter_T'][lamda] = []
                    for interT in self.inter_T[lamda]:
                        if round_zeros:
                            interT.set_to_zeroes(tol)
                        save_dict['inter_T'][lamda].append(interT.to_dict())
    
                text = "sparse stoch trans mats"
                
                
            if compressed:
                
                print('PID ', os.getpid(), ' : ',' saving ' + text + ' to ' + file)
                
                with gzip.open(file, 
                               'wb', compresslevel=2) as fopen:
                    pickle.dump(save_dict, fopen) 
            else:

                print('PID ', os.getpid(), ' : ',' saving ' + text + ' to ' + file)
                
                with open(file, 'wb') as fopen:
                    pickle.dump(save_dict, fopen)
                    
        

    
    @staticmethod 
    def load_inter_T(filename):
        """ loads inter_T and inter_T_lin from 'filename' that was saved with 
            save_inter_T.
            
            returns a dictionary with the inter_T restored.
            
        """
        
        ext = os.path.splitext(filename)[-1]
            
    
        if ext not in ['.pickle','.gz']:
            # detect extension
            if os.path.exists(filename + '.pickle'):
                ext = '.pickle'
                filename += '.pickle'
            elif os.path.exists(filename + '.gz'):
                ext = '.gz'
                filename += '.gz'
            elif os.path.exists(filename + '.pickle.gz'):
                ext = '.pickle.gz'
                filename += '.pickle.gz'
            else:
                raise FileNotFoundError(filename)
                
        if ext == '.gz' or ext == '.pickle.gz':
            with gzip.open(filename, 
                           'rb') as fopen:
                load_dict = pickle.load(fopen)
            
        else:
            with open(filename, 'rb') as fopen:
                load_dict = pickle.load(fopen)
                
        return_dict = {'_k_start_laplacians' : load_dict['_k_start_laplacians'], 
                       '_k_stop_laplacians' : load_dict['_k_stop_laplacians'], 
                       '_t_start_laplacians' : load_dict['_t_start_laplacians'], 
                       '_t_stop_laplacians' : load_dict['_t_stop_laplacians'],
                       'num_nodes' : load_dict['num_nodes'],
                       'times_k_start_to_k_stop+1' : load_dict['times_k_start_to_k_stop+1']}
                
        # rebuild inter_T from delta_inter_T
        if 'inter_T' in load_dict.keys():
            return_dict['inter_T'] = dict()
            
            if load_dict.get('is_sparse_stoch', False):
                
                for lamda in load_dict['inter_T'].keys():
                    return_dict['inter_T'][lamda] = \
                        [sparse_stoch_mat(**mat_dict) for mat_dict in \
                                          load_dict['inter_T'][lamda]]
        
            else:
                for lamda in load_dict['inter_T'].keys():
                    return_dict['inter_T'][lamda] = \
                        [load_dict['inter_T'][lamda]['trans_mat0']]
                        
                    for dT in load_dict['inter_T'][lamda]['delta_inter_T']:
                        return_dict['inter_T'][lamda].append(\
                                 return_dict['inter_T'][lamda][-1] + dT)
            
            
        if 'inter_T_lin' in load_dict.keys():
            return_dict['inter_T_lin'] = dict()
            
            if load_dict.get('is_sparse_stoch', False):
                
                for lamda in load_dict['inter_T_lin'].keys():
                    return_dict['inter_T_lin'][lamda] = dict()
                    
                    for t_s in load_dict['inter_T_lin'][lamda].keys():
    
                        return_dict['inter_T_lin'][lamda][t_s] = \
                            [sparse_stoch_mat(**mat_dict) for mat_dict in \
                                              load_dict['inter_T_lin'][lamda][t_s]]
            
            else:               
                for lamda in load_dict['inter_T_lin'].keys():
                    return_dict['inter_T_lin'][lamda] = dict()
                    
                    for t_s in load_dict['inter_T_lin'][lamda].keys():
    
                        return_dict['inter_T_lin'][lamda][t_s] = \
                            [load_dict['inter_T_lin'][lamda][t_s]['trans_mat_lin0']]
                            
                        for dT in load_dict['inter_T_lin'][lamda][t_s]['delta_inter_T_lin']:
                            return_dict['inter_T_lin'][lamda][t_s].append(\
                                     return_dict['inter_T_lin'][lamda][t_s][-1]+dT)
                


        del load_dict
        
        return return_dict            
                   
    def save_T(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                                   compressed=False):
        """ 
            saves a dict with 'T' as key and net.T as item with other 
            useful attributes.
            
            also works with sparse_stoch_mat.
            
            only works if net.T[lamda] is a matrix and not a list of matrices,
            i.e. if compute_transition_matrices was ran without save_intermediate.
                        
        """
        assert hasattr(self, 'T'), f'PID {os.getpid()} : nothing saved, compute inter_T first.'
        
        save_dict = {}
        save_dict['_k_start_laplacians'] = self._k_start_laplacians
        save_dict['_k_stop_laplacians'] = self._k_stop_laplacians
        save_dict['_t_start_laplacians'] = self._t_start_laplacians
        save_dict['_t_stop_laplacians'] = self._t_stop_laplacians
        save_dict['times_k_start_to_k_stop+1'] = self.times.values[\
                            self._k_start_laplacians:self._k_stop_laplacians+1]
        save_dict['num_nodes'] = self.num_nodes
        save_dict['_compute_times'] = self._compute_times
        

        
        save_dict['T'] = dict()
    
        if lamda is None:
            lamdas = self.T.keys()
        else:
            lamdas = [lamda]
        
        for lamda in lamdas:
            if isinstance(self.T[lamda], sparse_stoch_mat):
                save_dict['is_sparse_stoch'] = True
                if round_zeros:
                    T = self.T[lamda].copy()
                    T.set_to_zeroes(tol)
                else:
                    T = self.T[lamda]
                save_dict['T'][lamda] = T.to_dict()
                
                text = 'sparse_stoch_mat T'
                    
            elif isspmatrix_csr(self.T[lamda]):
                
                if round_zeros:
                    T = self.T[lamda].copy()
                    set_to_zeroes(T, tol)
                else:
                    T = self.T[lamda]
                save_dict['T'][lamda] = T
                
                text = 'csr T'
                
            else:
                raise TypeError('T must be csr or sparse_stoch_mat.')
            
            
        ext = os.path.splitext(filename)[-1]

        file = filename
            
        if compressed:
                            
            if ext != '.gz':
                file += '.gz'
            
            print('PID ', os.getpid(), ' : ',' saving ' + text + ' to ' + file)
            
            with gzip.open(file, 
                           'wb', compresslevel=2) as fopen:
                pickle.dump(save_dict, fopen) 
        else:
            ext = os.path.splitext(filename)[-1]
            if ext != '.pickle':
                file += '.pickle'
            
            print('PID ', os.getpid(), ' : ',' saving ' + text + ' to ' + file)
            
            with open(file, 'wb') as fopen:
                pickle.dump(save_dict, fopen)
                
    def save_T_lin(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                                   compressed=False):
        """ 
            saves a dict with 'T_lin' as key and net.T_lin as item with other 
            useful attributes.
            
            also works with sparse_stoch_mat.
            
            only works if net.T_lin[lamda][t_s] is a matrix and not a list of matrices,
            i.e. if compute_transition_matrices was ran without save_intermediate.
                        
        """
        assert hasattr(self, 'T_lin'), f'PID {os.getpid()} : nothing saved, compute inter_T_lin first.'
        
        save_dict = {}
        save_dict['_k_start_laplacians'] = self._k_start_laplacians
        save_dict['_k_stop_laplacians'] = self._k_stop_laplacians
        save_dict['_t_start_laplacians'] = self._t_start_laplacians
        save_dict['_t_stop_laplacians'] = self._t_stop_laplacians
        save_dict['times_k_start_to_k_stop+1'] = self.times.values[\
                            self._k_start_laplacians:self._k_stop_laplacians+1]
        save_dict['num_nodes'] = self.num_nodes
        save_dict['_compute_times'] = self._compute_times

        
        save_dict['T_lin'] = dict()
    
        if lamda is None:
            lamdas = self.T_lin.keys()
        else:
            lamdas = [lamda]
        
        for lamda in lamdas:
            save_dict['T_lin'][lamda] = dict()
            for t_s in self.T_lin[lamda].keys():
                
                if isinstance(self.T_lin[lamda][t_s], sparse_stoch_mat):
                    save_dict['is_sparse_stoch'] = True
                    if round_zeros:
                        T = self.T_lin[lamda][t_s].copy()
                        T.set_to_zeroes(tol)
                    else:
                        T = self.T_lin[lamda][t_s]
                    
                    save_dict['T_lin'][lamda][t_s] = T.to_dict()
                    
                    text = 'sparse_stoch_mat T_lin'
                        
                elif isspmatrix_csr(self.T_lin[lamda][t_s]):
                    
                    if round_zeros:
                        T = self.T_lin[lamda][t_s].copy()
                        set_to_zeroes(T, tol)
                    else:
                        T = self.T_lin[lamda][t_s]
                    save_dict['T_lin'][lamda][t_s] = T
                    
                    text = 'csr T_lin'
                    
                else:
                    raise TypeError('T must be csr or sparse_stoch_mat.')
            
            
        ext = os.path.splitext(filename)[-1]

        file = filename
            
        if compressed:
                            
            if ext != '.gz':
                file += '.gz'
            
            print('PID ', os.getpid(), ' : ',' saving ' + text + ' to ' + file)
            
            with gzip.open(file, 
                           'wb', compresslevel=2) as fopen:
                pickle.dump(save_dict, fopen) 
        else:
            ext = os.path.splitext(filename)[-1]
            if ext != '.pickle':
                file += '.pickle'
            
            print('PID ', os.getpid(), ' : ',' saving ' + text + ' to ' + file)
            
            with open(file, 'wb') as fopen:
                pickle.dump(save_dict, fopen)                
                
    @staticmethod 
    def load_T(filename):
        """ loads T and T_lin from 'filename' that was saved with 
            save_T.
            
            returns a dictionary with the T restored.
            
        """
        
        ext = os.path.splitext(filename)[-1]
            
    
        if ext not in ['.pickle','.gz']:
            # detect extension
            if os.path.exists(filename + '.pickle'):
                ext = '.pickle'
                filename += '.pickle'
            elif os.path.exists(filename + '.gz'):
                ext = '.gz'
                filename += '.gz'
            elif os.path.exists(filename + '.pickle.gz'):
                ext = '.pickle.gz'
                filename += '.pickle.gz'
            else:
                raise FileNotFoundError(filename)
                
        if ext == '.gz' or ext == '.pickle.gz':
            with gzip.open(filename, 
                           'rb') as fopen:
                load_dict = pickle.load(fopen)
            
        else:
            with open(filename, 'rb') as fopen:
                load_dict = pickle.load(fopen)
                
        return_dict = {'_k_start_laplacians' : load_dict['_k_start_laplacians'], 
                       '_k_stop_laplacians' : load_dict['_k_stop_laplacians'], 
                       '_t_start_laplacians' : load_dict['_t_start_laplacians'], 
                       '_t_stop_laplacians' : load_dict['_t_stop_laplacians'],
                       'num_nodes' : load_dict['num_nodes'],
                       'times_k_start_to_k_stop+1' : load_dict['times_k_start_to_k_stop+1']}
                
        
        if 'T' in load_dict.keys():
            return_dict['T'] = dict()
            
            if load_dict.get('is_sparse_stoch', False):
                
                for lamda in load_dict['T'].keys():
                    return_dict['T'][lamda] = \
                        sparse_stoch_mat(**load_dict['T'][lamda])
        
            else:
                for lamda in load_dict['T'].keys():
                    return_dict['T'][lamda] = load_dict['T'][lamda]
            
            
        if 'T_lin' in load_dict.keys():
            return_dict['T_lin'] = dict()
            
            if load_dict.get('is_sparse_stoch', False):
                
                for lamda in load_dict['T_lin'].keys():
                    return_dict['T_lin'][lamda] = dict()
                    
                    for t_s in load_dict['T_lin'][lamda].keys():
    
                        return_dict['T_lin'][lamda][t_s] = \
                            sparse_stoch_mat(**load_dict['T_lin'][lamda][t_s])
            
            else:               
                for lamda in load_dict['T_lin'].keys():
                    return_dict['T_lin'][lamda] = dict()
                    
                    for t_s in load_dict['T_lin'][lamda].keys():
    
                        return_dict['T_lin'][lamda][t_s] = \
                                            load_dict['T_lin'][lamda][t_s]
                            


        del load_dict
        
        return return_dict                            
                
    
    def compute_static_adjacency_matrix(self,
                               start_time=None,
                               end_time=None):
        """
        Returns the adjacency matrix of the static network built from the 
        aggregagted edge activity between `start_time` and `end_time`.

        Parameters
        ----------
        start_time : float or int, optional
            Starting time for the aggregation. The default is None, i.e. the 
            start time of the entire temporal network.
        end_time : float or int, optional
            Ending time for the aggregation. The default is None, i.e. the 
            end time of the entire temporal network.

        Returns
        -------
        CSR sparse matrix
            Symmetric adjacency matrix, where element ij is equal to the 
            aggregated time during which egde ij was active after `start_time` 
            and before `end_time`.

        """
        
        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time
            
        mask = np.logical_and(self.events_table.starting_times < end_time,
                              self.events_table.ending_times > start_time)
        
        # loop on events
        data = []
        cols = []
        rows = []
        for ev in self.events_table.loc[mask].itertuples():
            data.append(min(ev.ending_times, end_time) - max(ev.starting_times, start_time))
            rows.append(ev.source_nodes)
            cols.append(ev.target_nodes)
            
        A = coo_matrix((data, (rows,cols)), 
                       shape=(self.num_nodes, self.num_nodes))
        
        return A + A.T

    
    def _compute_time_grid(self):
        """ create `self.time_grid`, a dataframe with ('times', 'id') as index,
            were `id` is the index of the corresponding event in `events_table`,
            and column 'is_start' which is True is the ('times', 'id') 
            corresponds to a starting event.
            Also creates `self.times`, an array with all the times values.
            
        """
        self.time_grid = pd.DataFrame(columns=['times','id','is_start'],
                                 index=range(self.events_table.shape[0]*2))
        self.time_grid.iloc[:self.events_table.shape[0],[0,1]] = \
                     self.events_table.reset_index()[['starting_times','index']]
        self.time_grid['is_start'] = False
        self.time_grid.loc[0:self.events_table.shape[0]-1,'is_start'] = True
        
        self.time_grid.iloc[self.events_table.shape[0]:,[0,1]] = \
                      self.events_table.reset_index()[['ending_times','index']]
                      
        self.time_grid.times = pd.to_numeric(self.time_grid.times)
        
        self.time_grid.sort_values('times', inplace=True)
        
        # group events with same times
        self.time_grid.set_index(['times', 'id'], inplace=True)
        
        self.time_grid.sort_index(inplace=True)
        
        self.times = self.time_grid.index.levels[0]
        
        
    def _get_closest_time(self, t):
        """ return closest smaller or equal time in `times` and its index"""
        
        if not hasattr(self, 'times'):
            self._compute_time_grid()
            
        if t not in self.times:
        # take the largest smaller time
            if t <= self.times[0]:
                t = self.times[0]
            else:
                t = self.times[self.times <= t].max()
            
        k = int(np.where(self.times == t)[0])
        
        return t, k
        
            
    def compute_laplacian_matrices(self, t_start=None,
                                       t_stop=None, verbose=False,
                                       save_adjacencies=False,
                                       random_walk=True):
        """ Computes the laplacian matrices and saves them in `self.laplacians`
            
            Computes from the first event time (in `self.times`) before or equal to `t_start` until
            the event time index before `t_stop`.
            
            Laplacians are computed from `self.times[self._k_start_laplacians]`
            until `self.times[self._k_stop_laplacians-1]`.
            
            The laplacian at step k, is the random walk laplacian
            between `times[k]`. and `times[k+1]`.
            
        Parameters
        ----------
        t_start : float or int, optional
            The default is None, i.e. starts at the beginning of times.
            The computation starts from the first time index before or equal to t_start.
            The corresponding starting time index is saved in `self._k_start_laplacians`
            and the real starting time is `self.times[self._k_start_laplacians]` which 
            is saved in `self._t_start_laplacians`.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations. Default is end of times.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Similarily to `t_start`, the corresponding times are saved in 
            `self._k_stop_laplacians` and `self._t_stop_laplacians`.
        verbose : bool, optional
            The default is False.
        save_adjacencies : bool, optional
            Default is False. Use to save adjacency matrices in `self.adjacencies`.

        Returns
        -------
        None.            
        """
            
        if verbose:
            print('PID ', os.getpid(), ' : ','Computing Laplacians')
            
        if not hasattr(self, 'time_grid'):
            self._compute_time_grid()
            
        # instantaneous adjacency matrix
        A = lil_matrix((self.num_nodes, self.num_nodes),
                       dtype=np.float64)
        
        #identity 
        I = eye(self.num_nodes,
                dtype=np.float64).tocsc()
        
        #degree array
        degrees = np.zeros(self.num_nodes, dtype=np.float64)
        # inverse degrees diagonal matrix
        Dm1 = I.copy()
        # self loop matrix
        S = I.copy()
        
    
        self.laplacians = []
        if save_adjacencies:
            self.adjacencies = []

        # set boundary conditions : L_k is the laplacian during t_k and t_k+1
        if t_start is None:
            self._t_start_laplacians = self.times[0]
            self._k_start_laplacians = 0
        else:
            t, k = self._get_closest_time(t_start)
            self._t_start_laplacians = t
            self._k_start_laplacians = k
        if t_stop is None:
            self._t_stop_laplacians = self.times[-1]
            self._k_stop_laplacians = len(self.times)-1
        else:
            t, k = self._get_closest_time(t_stop)
            self._t_stop_laplacians = t
            self._k_stop_laplacians = k
                                    
        t0 = time.time()

        if random_walk is True:
            if self._k_start_laplacians > 0:
                # initial conditions, we have to find the adjacency mat just before
                # _k_start_laplacians. 
                
                t_km1 = self.times[self._k_start_laplacians-1]
                
                # find events that have started before or at t_k-1
                # and were still occuring at t_k-1
                mask_ini = (self.events_table.starting_times <= \
                                                    t_km1) & \
                            (self.events_table.ending_times > \
                                                    t_km1)
                
                for event in self.events_table.loc[mask_ini][['source_nodes',
                                            'target_nodes']].itertuples():
                    
                    if A[event.source_nodes, event.target_nodes] != 1:
                            A[event.source_nodes, event.target_nodes] = 1
                            A[event.target_nodes, event.source_nodes] = 1
                            
                            degrees[event.source_nodes] += 1
                            degrees[event.target_nodes] += 1
                    
                    if degrees[event.source_nodes] == 0:
                            S.data[event.source_nodes] = 1
                            Dm1.data[event.source_nodes] = 1
                    else:
                        S.data[event.source_nodes] = 0
                        Dm1.data[event.source_nodes] = 1/degrees[event.source_nodes]
                        
                    if degrees[event.target_nodes] == 0:
                        S.data[event.target_nodes] = 1
                        Dm1.data[event.target_nodes] = 1
                    else:
                        S.data[event.target_nodes] = 0
                        Dm1.data[event.target_nodes] = 1/degrees[event.target_nodes]


            # time grid for this time range            
            time_grid_range = self.time_grid.loc[\
                    (self.time_grid.index.get_level_values('times') >= \
                        self._t_start_laplacians) & \
                    (self.time_grid.index.get_level_values('times') < \
                        self._t_stop_laplacians)]
                        
            for k, (tk, time_ev) in enumerate(time_grid_range.groupby(level='times')):
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , 
                        self._k_stop_laplacians - self._k_start_laplacians)
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                                
                meet_id = time_ev.index.get_level_values('id')
                # starting or ending events
                is_starts = time_ev.is_start.values
                
                
                events_k = [self.events_table.loc[mid,['source_nodes','target_nodes']].astype(np.int64) \
                                            for mid in meet_id.values]            
                
                #update instantaneous matrices
                for event, is_start in zip(events_k, is_starts):
                    # unweighted, undirected
                    if is_start:
                        # if they are not already connected (can happen if the 
                        # opposite event overlap)
                        if A[event.source_nodes, event.target_nodes] != 1:
                            A[event.source_nodes, event.target_nodes] = 1
                            A[event.target_nodes, event.source_nodes] = 1
                            
                            degrees[event.source_nodes] += 1
                            degrees[event.target_nodes] += 1
                            
                    else:
                        #end of meeting
                        # if they are already unconnected
                        if A[event.source_nodes, event.target_nodes] > 0:
                            A[event.source_nodes, event.target_nodes] = 0
                            A[event.target_nodes, event.source_nodes] = 0
                            
                            degrees[event.source_nodes] -= 1
                            degrees[event.target_nodes] -= 1
                
                    # update self loops        
                    if degrees[event.source_nodes] == 0:
                        S.data[event.source_nodes] = 1
                        Dm1.data[event.source_nodes] = 1
                    else:
                        S.data[event.source_nodes] = 0
                        Dm1.data[event.source_nodes] = 1/degrees[event.source_nodes]
                        
                    if degrees[event.target_nodes] == 0:
                        S.data[event.target_nodes] = 1
                        Dm1.data[event.target_nodes] = 1
                    else:
                        S.data[event.target_nodes] = 0
                        Dm1.data[event.target_nodes] = 1/degrees[event.target_nodes]
                
                # Laplacian L(tk)
                Acsc = A.tocsc()
                T_D = Dm1 @ (Acsc + S)
                L = I - T_D
            
                self.laplacians.append(I - Dm1 @ (Acsc + S)) #(I - Acsc)
                if save_adjacencies:
                    self.adjacencies.append(A.copy())
        else: #case random_walk is False
            if self._k_start_laplacians > 0:
                # initial conditions, we have to find the adjacency mat just before
                # _k_start_laplacians. 
                
                t_km1 = self.times[self._k_start_laplacians-1]
                
                # find events that have started before or at t_k-1
                # and were still occuring at t_k-1
                mask_ini = (self.events_table.starting_times <= \
                                                    t_km1) & \
                            (self.events_table.ending_times > \
                                                    t_km1)
                
                for event in self.events_table.loc[mask_ini][['source_nodes',
                                            'target_nodes']].itertuples():
                    
                    if A[event.source_nodes, event.target_nodes] != 1:
                            A[event.source_nodes, event.target_nodes] = 1
                            A[event.target_nodes, event.source_nodes] = 1
                            
                            degrees[event.source_nodes] += 1
                            degrees[event.target_nodes] += 1
                    
                    if degrees[event.source_nodes] == 0:
                            Dm1.data[event.source_nodes] = 1
                    else:
                        Dm1.data[event.source_nodes] = degrees[event.source_nodes]
                        
                    if degrees[event.target_nodes] == 0:
                        Dm1.data[event.target_nodes] = 1
                    else:
                        Dm1.data[event.target_nodes] = degrees[event.target_nodes]


            # time grid for this time range            
            time_grid_range = self.time_grid.loc[\
                    (self.time_grid.index.get_level_values('times') >= \
                        self._t_start_laplacians) & \
                    (self.time_grid.index.get_level_values('times') < \
                        self._t_stop_laplacians)]
                        
            for k, (tk, time_ev) in enumerate(time_grid_range.groupby(level='times')):
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , 
                        self._k_stop_laplacians - self._k_start_laplacians)
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                                
                meet_id = time_ev.index.get_level_values('id')
                # starting or ending events
                is_starts = time_ev.is_start.values
                
                
                events_k = [self.events_table.loc[mid,['source_nodes','target_nodes']].astype(np.int64) \
                                            for mid in meet_id.values]            
                
                #update instantaneous matrices
                for event, is_start in zip(events_k, is_starts):
                    # unweighted, undirected
                    if is_start:
                        # if they are not already connected (can happen if the 
                        # opposite event overlap)
                        if A[event.source_nodes, event.target_nodes] != 1:
                            A[event.source_nodes, event.target_nodes] = 1
                            A[event.target_nodes, event.source_nodes] = 1
                            
                            degrees[event.source_nodes] += 1
                            degrees[event.target_nodes] += 1
                            
                    else:
                        #end of meeting
                        # if they are already unconnected
                        if A[event.source_nodes, event.target_nodes] > 0:
                            A[event.source_nodes, event.target_nodes] = 0
                            A[event.target_nodes, event.source_nodes] = 0
                            
                            degrees[event.source_nodes] -= 1
                            degrees[event.target_nodes] -= 1
                
                    # update self loops        
                    if degrees[event.source_nodes] == 0:
                        Dm1.data[event.source_nodes] = 0
                    else:
                        Dm1.data[event.source_nodes] = degrees[event.source_nodes]
                        
                    if degrees[event.target_nodes] == 0:
                        Dm1.data[event.target_nodes] = 0
                    else:
                        Dm1.data[event.target_nodes] = degrees[event.target_nodes]
                
                # Laplacian L(tk)
                Acsc = A.tocsc()
                L = Dm1 - Acsc
            
                self.laplacians.append(Dm1 - Acsc)
                if save_adjacencies:
                    self.adjacencies.append(A.copy())
            
        t_end = time.time()-t0
        self._compute_times['laplacians'] = t_end
        if verbose:
            print('PID ', os.getpid(), ' : ','finished in ', t_end)            
        
    def compute_inter_transition_matrices(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    fix_tau_k=False,
                                    use_sparse_stoch=False,
                                    dense_expm=True,
                                    random_walk=True):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose, random_walk=random_walk)
        if not hasattr(self, 'inter_T'):
            self.inter_T = dict()            
    
        
        if lamda is None:
            if verbose:
                print('PID ', os.getpid(), ' : ','taking lamda as 1/tau_w with tau_w = median interevent time')
            lamda = 1/np.median(np.diff(self.times))
        
        # new value of lamda, we need to recompute
        if lamda not in self.inter_T.keys():
            
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing interevent transition matrices for lamda={0}'.format(lamda))

            self.inter_T[lamda] = []
                
            t0 = time.time()
            for k, tk in enumerate(self.times[self._k_start_laplacians:
                                              self._k_stop_laplacians]):
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' ,
                          self._k_stop_laplacians-1-self._k_start_laplacians)
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                if fix_tau_k:
                    tau_k = 1.0
                else:
                    tau_k = self.times[self._k_start_laplacians+k+1] - tk
                
                if use_sparse_stoch:
                    self.inter_T[lamda].append(sparse_lapl_expm(self.laplacians[k],
                                                                       tau_k*lamda,
                                                                       dense_expm=dense_expm))
                else:
                    if self.laplacians[k].getnnz() == 0:
                        # expm of zero = identity
                        self.inter_T[lamda].append(eye(self.num_nodes,format='csr'))
                    else:
                        if dense_expm:
                            self.inter_T[lamda].append(csr_matrix(expm(-tau_k*lamda*self.laplacians[k].toarray())))
                        else:
                            self.inter_T[lamda].append(expm(-tau_k*lamda*self.laplacians[k]).tocsr())
        
            if len(self.inter_T[lamda]) == 0:
                if verbose:
                    print('PID ', os.getpid(), ' no events, trans. matrix = identity')
                # is there was no event, the transition is identity
                if use_sparse_stoch:
                    self.inter_T[lamda].append(sparse_stoch_mat.create_diag(size=self.num_nodes))
                else:
                    self.inter_T[lamda].append(eye(self.num_nodes, 
                                                      dtype=np.float64,
                                                      format='csr'))
                
            t_end = time.time()-t0
            
            self._compute_times['inter_T_' + str(lamda)]  = t_end
            
            if verbose:
                print('PID ', os.getpid(), ' : ','Finished computing interevent transition matrices in ', t_end)
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Interevent transition matrices already computed for lamda={0}'.format(lamda))
            
    
            
    def compute_transition_matrices(self, lamda=None,
                                    verbose=False,
                                    save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=False,
                                    tol=None):
        """ compute transition matrices and saves them in a dict of lists `self.T[lamda]` 
            where `self.T[lamda][k]` is the product of all interevent transition 
            matrices from t_0 to t_k computed with lamda.
        """
            
        if not hasattr(self, 'inter_T') or \
                lamda not in self.inter_T.keys():
            raise Exception("Compute inter_T first.")

            
        if not hasattr(self, 'T'):
            self.T = dict()
            
        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))
            
            
        if lamda not in self.T.keys():
            if save_intermediate:
                if force_csr:
                    # forcing the first matrix to csr, will ensure that 
                    # all products are done in csr format,
                    # since CSR @ SparseStochMat t is not implemented
                    self.T[lamda] = [self.inter_T[lamda][k_init].tocsr()]
                else:
                    self.T[lamda] = [self.inter_T[lamda][k_init]]
                
                if tol is not None:
                    set_to_zeroes(self.T[lamda][0],tol)
                    inplace_csr_row_normalize(self.T[lamda][0])
            else:
                if force_csr:
                    self.T[lamda] = self.inter_T[lamda][k_init].tocsr()
                else:
                    self.T[lamda] = self.inter_T[lamda][k_init]
                    
                if tol is not None:
                    set_to_zeroes(self.T[lamda],tol)
                    inplace_csr_row_normalize(self.T[lamda])
                    
                    
            
        
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing transition matrix')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                Tk = self.inter_T[lamda][k]
                if tol is not None:
                    set_to_zeroes(Tk,tol)
                    inplace_csr_row_normalize(Tk)
                    
                if save_intermediate:
                    
                    self.T[lamda].append(self.T[lamda][-1] @ Tk)
                    
                    #normalize T to correct precision errors
                    if tol is not None:
                        set_to_zeroes(self.T[lamda][-1],tol)
                        
                    inplace_csr_row_normalize(self.T[lamda][-1])
                else:
                    self.T[lamda] = self.T[lamda] @ Tk
                    if tol is not None:
                        set_to_zeroes(self.T[lamda], tol)
                        
                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T[lamda])
                
            t_end = time.time()-t0
            
            self._compute_times['trans_matrix_{0}_rev{1}'.format(lamda,reverse_time)] = t_end
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Transition matrices already computed for lamda={0}'.format(lamda))      

    def compute_transition_matrices_sliding_window(self, lamda=None,
                                                verbose=False,
                                                save_intermediate=True,
                                                reverse_time=False,
                                                force_csr=False,
                                                tol=None,
                                                window_size=None):
        """ Compute transition matrices using a sliding window approach.
            Saves them in a dict of lists `self.T[lamda]` where `self.T[lamda][k]`
            is the product of interevent transition matrices from t_{k-window_size} to t_k.
        
            Parameters:
            - lamda: parameter for interevent matrices.
            - t_start, t_stop: time range for computation.
            - verbose: flag for detailed logging.
            - save_intermediate: whether to save intermediate results.
            - reverse_time: if True, computes in reverse time.
            - force_csr: forces computation in CSR format.
            - tol: tolerance for setting matrix elements to zero.
            - window_size: size of the sliding window (must be provided).
        """

        if not hasattr(self, 'inter_T') or lamda not in self.inter_T.keys():
            raise Exception("Compute inter_T first.")

        if not hasattr(self, 'window_T'):
            self.window_T = dict()

        if window_size is None or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        if reverse_time:
            k_init = len(self.inter_T[lamda]) - 1
            k_range = reversed(range(window_size - 1, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(window_size - 1, len(self.inter_T[lamda]))

        if lamda not in self.window_T.keys():
            self.window_T[lamda] = []

        if verbose:
            print('PID ', os.getpid(), ' : ', 'Computing sliding window transition matrices')

        t0 = time.time()

        for k in k_range:
            if verbose and not k % 1000:
                print('PID ', os.getpid(), ' : ', k, ' over ', len(self.inter_T[lamda]))
                print(f'PID {os.getpid()} : {time.time() - t0:.2f}s')

            # Compute product over the sliding window
            Tk_window = self.inter_T[lamda][k - window_size + 1]
            for i in range(k - window_size + 2, k + 1):
                if tol is not None:
                    set_to_zeroes(self.inter_T[lamda][i], tol)
                    inplace_csr_row_normalize(self.inter_T[lamda][i])
                Tk_window = Tk_window @ self.inter_T[lamda][i]

            if tol is not None:
                set_to_zeroes(Tk_window, tol)
                inplace_csr_row_normalize(Tk_window)

            if save_intermediate:
                self.window_T[lamda].append(Tk_window)

        t_end = time.time() - t0

        if verbose:
            print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')


    def compute_entropy(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    tol=None):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'S'):
            self.S = dict()

        
        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))
            
        p0 = 1/self.num_nodes*np.ones(self.num_nodes)
        
        if lamda not in self.S.keys():
            #if save_intermediate:
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                p = p0 @ self.T[lamda][k_init].tocsr()
                T = self.T[lamda][k_init].tocsr()
                logTdata = np.log(np.where(T.data > 0, T.data, 1))
                TlogT = csr_matrix((T.data*logTdata, T.indices, T.indptr), shape=T.shape)
                # there shouldn't be need for this
                # TlogT[TlogT>0]=0
                self.S[lamda] = [-np.sum(p0 @ TlogT, where= np.isfinite(p0 @ TlogT))]
            else:
                raise Exception("Use force_csr=True")
            
            if tol is not None:
                set_to_zeroes(self.S[lamda][0],tol)
                # inplace_csr_row_normalize(self.T[lamda][0])
            '''else:
                if force_csr:
                    self.T[lamda] = self.inter_T[lamda][k_init].tocsr()
                else:
                    self.T[lamda] = self.inter_T[lamda][k_init]
                    
                if tol is not None:
                    set_to_zeroes(self.T[lamda],tol)
                    inplace_csr_row_normalize(self.T[lamda])'''
                    
                    
            
        
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                p = p0 @ self.T[lamda][k]
                T = self.T[lamda][k].tocsr()
                logTdata = np.log(np.where(T.data > 0, T.data, 1))
                TlogTdata = T.data * logTdata
                # there shouldn't be need for this
                # TlogT[TlogT>0]=0
                TlogT = csr_matrix((TlogTdata, T.indices, T.indptr), shape=T.shape)
                self.S[lamda].append(-np.sum(p0 @ TlogT, where= np.isfinite(p0 @ TlogT)))
                
            t_end = time.time()-t0
            
            self._compute_times['entropy_{0}_rev{1}'.format(lamda,reverse_time)] = t_end
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda)) 

    def compute_vonNeumann_entropy(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    time_domain=None):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'vNS'):
            self.vNS = dict()

        
        if time_domain is None:
            time_domain = self.times()
        if reverse_time:
            k_init = len(time_domain)-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(time_domain))
            
        
        if lamda not in self.vNS.keys():
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                T = self.T[lamda][k_init].tocsr()
                T = T.toarray()
                rho = T/np.trace(T)
                rhologrho = rho @ d_logm(rho)
                self.vNS[lamda] = [- 1/np.log(self.num_nodes) * np.trace(rhologrho)]
                del T
                del rho
                del rhologrho
            else:
                raise Exception
         
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                T = self.T[lamda][k].tocsr()
                T = T.toarray()
                rho = T/np.trace(T)
                rhologrho = rho @ d_logm(rho)
                self.vNS[lamda].append(- 1/np.log(self.num_nodes) * np.trace(rhologrho))
                del T
                del rho
                del rhologrho
            t_end = time.time()-t0
              
            
            self._compute_times['von Neumman entropy_{0}_rev{1}'.format(lamda,reverse_time)] = t_end
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda)) 

    def  compute_simple_vonNeumann_entropy(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'simple_vNS'):
            self.simple_vNS = dict()

        
        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))
            
        
        if lamda not in self.simple_vNS.keys():
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                T = self.T[lamda][k_init].tocsr()
                rho = T/T.trace()
                np_rhologrho = rho.data * np.log(rho.data)
                np_rhologrho[np.isnan(np_rhologrho)] = 0
                rhologrho = csr_matrix((np_rhologrho, rho.indices, rho.indptr), shape=rho.shape)
                del np_rhologrho
                self.simple_vNS[lamda] = [-rhologrho.trace()]
            else:
                raise Exception("Use force_csr=True")
                '''p = p0 @ self.T[lamda][k_init]
                logT = np.log(self.T[lamda][k_init])
                TlogT = np.multiply(self.T[lamda][k_init], logT)
                self.rho[lamda] = [np.sum(p @ TlogT)]'''
         
        
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                T = self.T[lamda][k].tocsr()
                rho = T/T.trace()
                np_rhologrho = rho.data * np.log(rho.data)
                np_rhologrho[np.isnan(np_rhologrho)] = 0
                rhologrho = csr_matrix((np_rhologrho, rho.indices, rho.indptr), shape=rho.shape)
                del np_rhologrho
                self.simple_vNS[lamda].append(-rhologrho.trace())

                
            t_end = time.time()-t0
            
            self._compute_times['von Neumman entropy_{0}_rev{1}'.format(lamda,reverse_time)] = t_end
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda))

    def compute_entropy2(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'S2'):
            self.S2 = dict()

        
        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))
            
        p0 = 1/self.num_nodes*np.ones(self.num_nodes)
        
        if lamda not in self.S2.keys():
            #if save_intermediate:
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                p = p0 @ self.T[lamda][k_init].tocsr()
                diagp = csr_matrix(np.diag(p))
                T = self.T[lamda][k_init].tocsr()
                Q = -lamda*self.laplacians[k_init]
                diagpQ= diagp @ Q
                array_diagpQ = np.asarray(diagpQ.toarray())
                
                entropy2 = 0
                for i in range(array_diagpQ.shape[0]):
                    for j in range(diagpQ.shape[1]):
                        if np.isfinite(np.log(array_diagpQ[i][j]/array_diagpQ[j][i])):
                            entropy2 += array_diagpQ[i][j] * np.log(array_diagpQ[i][j]/array_diagpQ[j][i])
                self.S2[lamda] = [entropy2]
            else:
                raise Exception("Use force_csr=True")
  
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy2 ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                p = p0 @ self.T[lamda][k].tocsr()
                diagp = csr_matrix(np.diag(p))
                T = self.T[lamda][k].tocsr()
                Q = -lamda*self.laplacians[k]
                diagpQ= diagp @ Q
                array_diagpQ = np.asarray(diagpQ.toarray())
                
                entropy2 = 0
                for i in range(array_diagpQ.shape[0]):
                    for j in range(diagpQ.shape[1]):
                        if np.isfinite(np.log(array_diagpQ[i][j]/array_diagpQ[j][i])):
                            entropy2 += array_diagpQ[i][j] * np.log(array_diagpQ[i][j]/array_diagpQ[j][i])
                self.S2[lamda].append(entropy2)

                
            t_end = time.time()-t0
            
            self._compute_times['entropy_{0}_rev{1}'.format(lamda,reverse_time)] = t_end
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy2 already computed for lamda={0}'.format(lamda)) 
    
                                   
    def _compute_stationary_transition(self, t_start=None, 
                                               t_stop=None, 
                                               verbose=False,
                                               use_sparse_stoch=True):
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start,
                                            t_stop=t_stop,
                                            verbose=verbose)
        if verbose:
            print('PID ', os.getpid(), ' : ','Computing stationary transition matrices')  
                
        self._stationary_trans = []
        
        I = eye(self.num_nodes,
                dtype=np.float64, format='csc')
        
    #   I = np.eye(self.num_nodes, dtype=np.float64)
                
        t0 = time.time()
        for k in range(len(self.laplacians)):
            if verbose and not k%1000:
                print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.laplacians))
                print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                
            if use_sparse_stoch:
                self._stationary_trans.append(\
                  sparse_stationary_trans(I - self.laplacians[k]))
            
            else:
                self._stationary_trans.append(\
                      compute_stationary_transition(I - self.laplacians[k]))
   
                
        t_end = time.time()-t0
        
        self._compute_times['_stationary_trans'] = t_end
        
        if verbose:
            print('PID ', os.getpid(), ' : ','Stationary transition matrices computation took {0}s'.format(t_end))
            
            

    
    def _merge_overlapping_events(self, verbose=False):
        """ merge temporally overlapping undirected event between each pair of nodes.
            
        """
        
        events_to_keep = np.ones(self.events_table.shape[0],dtype=bool)
        
        A = self.compute_static_adjacency_matrix()
                    
        # loop over nodes
        for i,n1 in enumerate(self.node_array):
            
            for n2 in (A[n1,:] > 0).nonzero()[1]:
                
                
                mask_12 = np.logical_and(self.events_table.source_nodes.values == n1, 
                                         self.events_table.target_nodes.values == n2)
                
                mask_21 = np.logical_and(self.events_table.source_nodes.values == n2, 
                                         self.events_table.target_nodes.values == n1)
                
                #sort by starting times
                evs = self.events_table.loc[np.logical_or(mask_12, mask_21)].\
                                sort_values(by=['starting_times', 'ending_times'])
                
                evs_list = list(evs.itertuples())
                
                # event to compare
                ev1 = evs_list[0]
                merged = 0
                for k in range(1,len(evs_list)):
                    ev2 = evs_list[k]
                    # if ev2 overlaps with ev1, merge them, otherwise ev2 becomes
                    # ev1
                    if ev2.starting_times < ev1.ending_times:
                        #merge
                        events_to_keep[ev2.Index] = False
                        self.events_table.loc[ev1.Index,'ending_times'] = ev2.ending_times
                        ev1._replace(ending_times=ev2.ending_times)
                        merged += 1
                    else:
                        ev1 = ev2
                if verbose:
                    print(f'n1,n2 ({n1},{n2}): {merged} merged')
                                    
        num_merged = (events_to_keep == False).sum()                        
        print('PID ', os.getpid(), ' : ','merged ', 
              num_merged, ' events')
                                
        self.events_table = self.events_table.loc[events_to_keep]
        
        self.events_table.reset_index(inplace=True, drop=True)
                                   
        self.num_nodes = self.node_array.shape[0]
        
        self.num_events = self.events_table.shape[0]
        
        self.start_time = self.events_table.starting_times.min()
    
        self.end_time = self.events_table.ending_times.max()
        
        self._compute_time_grid()
        
        return num_merged
    

                
        


    def _compute_delta_trans_mat(self, lamda, round_zeros=True, tol=1e-8):
        """ comptes and put in a attribute `delta_inter_T` the matrix differences 
            between each consecutive inter_T.
            
            self.delta_inter_T[lamda][k] = self.inter_T[lamda][k+1] - self.inter_T[lamda][k]
            
            The length of self.delta_inter_T[lamda] is len(self.inter_T[lamda]) - 1
            
        """
        
        if hasattr(self, 'inter_T') and lamda in self.inter_T.keys():
            
        
            if not hasattr(self, 'delta_inter_T'):
                self.delta_inter_T = dict()
    
            if lamda not in self.delta_inter_T.keys():
        
                self.delta_inter_T[lamda] = [self.inter_T[lamda][k+1] - \
                                                  self.inter_T[lamda][k] for k in \
                                                  range(len(self.inter_T[lamda])-1)]
                
                if round_zeros:
                    for M in self.delta_inter_T[lamda]:
                        set_to_zeroes(M, tol=tol)
                                           
            else:
                print('PID ', os.getpid(), ' : ',
                      'delta_inter_T has already been computed with lamda={0}'.format(lamda))
                
        else:
            print('PID ', os.getpid(), ' : ', 'delta_inter_T has not been computed')
            
        
        if hasattr(self, 'inter_T_lin') and lamda in self.inter_T_lin.keys():
            
        
            if not hasattr(self, 'delta_inter_T_lin'):
                self.delta_inter_T_lin = dict()
    
            if lamda not in self.delta_inter_T_lin.keys():
                
                self.delta_inter_T_lin[lamda] = dict()
                
                for t_s in self.inter_T_lin[lamda].keys():
                
                    self.delta_inter_T_lin[lamda][t_s] = [self.inter_T_lin[lamda][t_s][k+1] - \
                                                      self.inter_T_lin[lamda][t_s][k] for k in \
                                                      range(len(self.inter_T_lin[lamda][t_s])-1)]
                                      
                    if round_zeros:
                        for M in self.delta_inter_T_lin[lamda][t_s]:
                            set_to_zeroes(M, tol=tol)
                            
            else:
                print('PID ', os.getpid(), ' : ',
                      'delta_inter_T_lin has already been computed with lamda={0}'.format(lamda))
                
        else:
            print('PID ', os.getpid(), ' : ', 'delta_inter_T_lin has not been computed')
            
  
def compute_stationary_transition(T):
    """ compute the transition matrix at stationarity for matrix `T`

        Parameters:
        -----------
        
        T : scipy.sparse matrix or numpy.ndarray
            Transition matrix of the discrete time random walk.
            Can be a CSC or CSR matrix.

        Returns:
        --------
        
        Pi : scipy.sparse.csr matrix
            Stationary transition matrix.
            
    """
    
    num_nodes = T.shape[0]
    
    # otherwise 0 values may count as an edge
    T.eliminate_zeros()
    T.sort_indices()

    n_comp, comp_labels = connected_components(T,directed=False)
    comp_sizes = np.bincount(comp_labels)
    cmp_to_indices = {cmp : (comp_labels == cmp).nonzero()[0] for \
                        cmp in range(n_comp)}
    
    # constructors for sparse array
    data = np.zeros((comp_sizes**2).sum(), dtype=np.float64)
    indices = np.zeros((comp_sizes**2).sum(), dtype=np.int32)
    indptr = np.zeros(num_nodes+1, dtype=np.int32)
    
    # vector of degrees (number of nonzero elements of each row)
    if isspmatrix(T):
        degs = np.diff(T.indptr)
    else:
        degs = (T > 0).sum(1)
        
    data_ind = 0
    for row in range(num_nodes):
        cmp = comp_labels[row]
        cmp_degs = degs[cmp_to_indices[cmp]]
        data[data_ind:data_ind+comp_sizes[cmp]] = \
                                                cmp_degs/cmp_degs.sum()
        indices[data_ind:data_ind+comp_sizes[cmp]] = \
                        cmp_to_indices[cmp]
        indptr[row] = data_ind
        data_ind += comp_sizes[cmp]
    indptr[num_nodes] = data_ind
        
    # Stationary transition matrix
    
    return csr_matrix((data, indices, indptr), shape=(num_nodes,num_nodes), 
                    dtype=np.float64)
    
def compute_subspace_expm(A, n_comp=None, comp_labels=None, verbose=False,
                        thresh_ratio=None, 
                        normalize_rows=True,):
    """ compute the exponential matrix of `A` by applying expm on each connected
        subgraphs defined by A and recomposing it to return expm(A).

        Parameters:
        -----------
        
        A : scipy.sparse.csc_matrix
        
        thresh_ratio: float, optional.
            Threshold ratio used to trim negligible values in the resulting matrix.
            Values smaller than `max(expm(A))/thresh_ratio` are set to 
            zero. Default is None.
        normalize_rows: bool, optional.
            Whether rows of the resulting matrix are normalized to sum to 1.
        

        Returns:
        --------
        
        expm(A) : scipy.sparse.csr_matrix
            matrix exponential of A
            
    """
    
    num_nodes = A.shape[0]
    
    # otherwise 0 values may count as an edge
    A.eliminate_zeros()
    A.sort_indices()

    if (n_comp is None) or (comp_labels is None):
        n_comp, comp_labels = connected_components(A,directed=False)
    comp_sizes = np.bincount(comp_labels)
    cmp_indices = [(comp_labels == cmp).nonzero()[0] for \
                        cmp in range(n_comp)]
        
    if verbose:
        print('PID ', os.getpid(), f' : subspace_expm with {n_comp} components')
    
    # constructors for sparse array
    data = np.zeros((comp_sizes**2).sum(), dtype=np.float64)
    indices = np.zeros((comp_sizes**2).sum(), dtype=np.int32)
    indptr = np.zeros(num_nodes+1, dtype=np.int32)
            
    # if nproc == 1:
    #     expm_func = lambda M: expm(M)
    # else:
    #     expm_func = lambda M: compute_parallel_expm(M, nproc=nproc, 
    #                                                 thresh_ratio=None, 
    #                                                 normalize_rows=False,
    #                                                 verbose=verbose)
    subnets_expms = []
    for i, cmp_ind in enumerate(cmp_indices):
        if verbose:
            print('PID ', os.getpid(), f' : computing component {i} over {n_comp}, with size {cmp_ind.size}')
    
        subnets_expms.append(expm(A[cmp_ind,:][:,cmp_ind]).toarray())
    
    # reconstruct csr sparse matrix
    if verbose:
            print('PID ', os.getpid(), ' : reconstructing expm mat')
    data_ind = 0
    for row in range(num_nodes):
        cmp = comp_labels[row]
        cmp_expm = subnets_expms[cmp]
        sub_expm_row, = np.where(cmp_indices[cmp] == row)
        
        data[data_ind:data_ind+comp_sizes[cmp]] = cmp_expm[sub_expm_row,:]
        
        indices[data_ind:data_ind+comp_sizes[cmp]] = cmp_indices[cmp]
        
        indptr[row] = data_ind
        
        data_ind += comp_sizes[cmp]
        
    indptr[num_nodes] = data_ind
        
    expmA = csr_matrix((data, indices, indptr), shape=(num_nodes,num_nodes), 
                    dtype=np.float64)
    
    if thresh_ratio is not None:
        expmA.data[expmA.data<expmA.data.max()/thresh_ratio] = 0.0
        expmA.eliminate_zeros()
    if normalize_rows:
        inplace_csr_row_normalize(expmA)
    
    return expmA
    
            
def csc_row_normalize(X):
    """ row normalize scipy sparse csc matrices.
        returns a copy of X row-normalized and in CSC format.
    """
    
    X = X.tocsr()
    
    for i in range(X.shape[0]):
        row_sum = X.data[X.indptr[i]:X.indptr[i+1]].sum()
        if row_sum != 0:
            X.data[X.indptr[i]:X.indptr[i+1]] /= row_sum
    
    return X.tocsc()

def find_spectral_gap(L):
    """ L is assummed to be connected"""
    
    Lcsr = L.tocsr()
    
    I = eye(L.shape[0],
                dtype=np.float64,
                format='csr')
    
    degs = np.diff((I-Lcsr).indptr)
    
    D12 = diags(np.sqrt(degs),
                format='csr')
    Dm12 = diags(1/np.sqrt(degs),
                format='csr')
    
    Lsym = D12 @ Lcsr @ Dm12
    
    # stationary solution
    Pi = np.vstack([degs/degs.sum()]*L.shape[0])
    
    
    gap = eigsh(Lsym.toarray()-Pi,1,sigma=0, 
                return_eigenvectors=False)

    return gap


def remove_nnz_rowcol(L):
    """ returns a CSC or CSR matrix where the indices with zero row and column
        cols have been removed, also return an array of the indices of rows/columns
        with non-zero values and the (linear) size of L.
        
        
        Returns:
        -------
        L_small, nonzero_indices, size
        
    """
    # indicies with zero sum row AND col
    nonzerosum_rowcols = ~np.logical_and(L.getnnz(1)==0, 
                                        L.getnnz(0)==0)
    
    nonzero_indices, = (nonzerosum_rowcols).nonzero()
    
    return L[nonzerosum_rowcols][:,nonzerosum_rowcols], nonzero_indices, L.shape[0]

      
    
def numpy_rebuild_nnz_rowcol(T_data,
                            T_indices,
                            T_indptr,
                            zero_indices):
    """ returns a CSR matrix (data,indices,rownnz, shape) built from the CSR 
        matrix T_small but with
        added row-colums at zero_indicies (with 1 on the diagonal)
        
    
    """
    
    n_rows = T_indptr.size-1 + zero_indices.size

    data = np.zeros(T_data.size+zero_indices.size,
                                dtype=np.float64)
    indices = np.zeros(T_data.size+zero_indices.size,
                                dtype=np.int32)
    indptr = np.zeros(n_rows+1,
                                dtype=np.int32)
    new_col_inds = np.zeros(T_indptr.size-1,
                                dtype=np.int32)
    Ts_indices = np.zeros(T_indices.size,
                                dtype=np.int32)
    zero_set = set(zero_indices)
    

    # map col indices to new positions
    k = 0
    for i in range(n_rows):
        if i not in zero_set:
            new_col_inds[k] = i
            k +=1
            
        
                    
    for k,i in enumerate(T_indices):
        Ts_indices[k] = new_col_inds[i]
    
    row_id_small_t = -1
    data_ind = 0
    for row_id in range(n_rows):
        row_id_small_t +=1
        if row_id in zero_set:
            # add a row with just 1 on the diagonal
            data[data_ind] = 1.0
            indices[data_ind] = row_id
            indptr[row_id+1] = indptr[row_id]+1
            
            row_id_small_t -= 1
            data_ind += 1
            
        else:
            row_start = T_indptr[row_id_small_t]    
            row_end = T_indptr[row_id_small_t+1]  
            
            num_data_row = row_end - row_start
            
            data[data_ind:data_ind+num_data_row] = T_data[row_start:row_end]
            indices[data_ind:data_ind+num_data_row] = Ts_indices[row_start:row_end]
            indptr[row_id+1] = indptr[row_id]+num_data_row
            
            data_ind += num_data_row
            

    return (data, indices, indptr, n_rows)    


    
def set_to_ones(Tcsr, tol=1e-8):
    """ in place replaces ones in sparse matrix that are, within the tolerence, close
        to ones with actual ones
    """
    
    Tcsr.data[np.abs(Tcsr.data - 1) <= tol] = 1


def set_to_zeroes(Tcsr, tol=1e-8, relative=True, use_absolute_value=False):
    """ in place replaces zeroes in sparse matrix that are, within the tolerence, close
        to zero with actual zeroes.
        If tol is None, does nothing
    """
        
    if tol is not None:
        if isinstance(Tcsr, sparse_stoch_mat):
            Tcsr.set_to_zeroes(tol, relative=relative)
        elif isinstance(Tcsr, (csr_matrix,csc_matrix)):
            if Tcsr.data.size > 0:
                if relative:
                    # tol = tol*np.abs(Tcsr.data).max()
                    # finding the max of the absolute value without making a copy of the whole array
                    tol = tol*np.abs([Tcsr.data.min(),Tcsr.data.max()]).max()
                    
                if use_absolute_value:
                    Tcsr.data[np.abs(Tcsr.data) <= tol] = 0
                else:
                    Tcsr.data[Tcsr.data <= tol] = 0
                    
                Tcsr.eliminate_zeros()
        else:
            raise TypeError("Tcsr must be csc,csr or sparse_stoch_mat")
    

class StaticTempNetwork(object):
    """ Continuous time temporal network
    
    Parameters:
    -----------
    
        
    events_table: Pandas Dataframe
        Dataframe with columns 'source_nodes', 'target_nodes', 'starting_times'
        and 'ending_times' and index corresponding to event index. Used for
        instantiating a new ConTempNetwork from the event_table of an other one.
    
    
    """
    def __init__(self,
                       times=[],
                       adjacency=None,
                       #relabel_nodes=True,
                       #reset_event_table_index=True,
                       #node_to_label_dict=None,
                       #merge_overlapping_events=False,
                       #events_table=None)
    ):
        self.times = times
        self.adjacency = adjacency    
        self.num_nodes = self.adjacency.shape[0]
        
    def save(self, filename, 
             matrices_list=None,
             attributes_list=None):
        """ save graph event_table and matrices as pickle file
        
        Parameters:
        -----------
        
        filename: str
            Filename where to save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to save.
            The default list is:
                `matrices_list = ['laplacians',
                                    'adjacencies',
                                    'inter_T',
                                    'inter_T_lin',
                                    'T',
                                    'T_lin',
                                    '_stationary_trans',
                                    'delta_inter_T',
                                    'delta_inter_T_lin',]`
        attributes_list: list of strings
            List of attribute names to save.
            The default list is:
                `attributes_list = ['node_to_label_dict',
                                  'events_table',
                                  'times',
                                  'time_grid',
                                  'num_nodes',
                                  '_compute_times',
                                  '_t_start_laplacians',
                                  '_k_start_laplacians',
                                  '_t_stop_laplacians',
                                  '_k_stop_laplacians',
                                  '_overlapping_events_merged',]`
                                    
            
        """
        
        
        save_dict = dict()
        
        
        matrices = ['laplacians',
                      'inter_T',
                      'T']
        
        if matrices_list is None:
            matrices_list = matrices
        
        attributes = ['times',
                      'adjacency',
                      'num_nodes',
                      'S',
                      'vNS',
                      'simple_vNS',
                      'S2',
                      'new_vNS']
        
        if attributes_list is None:
            attributes_list = attributes
        
        for attr in attributes_list:
            if hasattr(self, attr):
                save_dict[attr] = getattr(self,attr)
                

        for mat in matrices_list:
            if hasattr(self, mat):
                save_dict[mat] = getattr(self,mat)
                

        with open(os.path.splitext(filename)[0] + '.pickle', 'wb') as fopen:
            pickle.dump(save_dict, fopen)
                        
        
    @classmethod 
    def load(cls, filename,
             matrices_list=None,
             attributes_list=None):
        """ load network from file and returns a ContTempNetwork instance.
        
        Parameters:
        -----------
        
        filename: str
            Filename where the network is saved save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to load.
            The default list is:
                `matrices_list = ['laplacians',
                      'inter_T',
                      'T']`
        attributes_list: list of strings
            List of attribute names to load.
            The default list is:
                `attributes_list = ['times',
                      'adjacency',
                      'num_nodes',
                      'S',
                      'vNS',
                      'simple_vNS',
                      'S2',
                      'new_vNS']`
                                    
            
            
        """
        
        matrices = ['laplacians',
                      'inter_T',
                      'T']
        
        if matrices_list is None:
            matrices_list = matrices
        
        attributes = ['times',
                      'adjacency',
                      'num_nodes',
                      'S',
                      'vNS',
                      'simple_vNS',
                      'S2',
                      'new_vNS'] 
                    
        if attributes_list is None:
            attributes_list = attributes
            
            
        # all in a pickle file

        graph_dict = pd.read_pickle(os.path.splitext(filename)[0] +'.pickle')

        net = cls(adjacency = graph_dict.pop('adjacency'))

        for k,val in graph_dict.items():
            if k in matrices_list:
                setattr(net, k, val)
            if k in attributes_list:
                setattr(net, k, val)
            
        return net

    def compute_laplacian_matrices(self, 
                                        verbose = False,
                                        random_walk = True):
        """ Computes the laplacian matrices and saves them in `self.laplacians`
            
            Computes from the first event time (in `self.times`) before or equal to `t_start` until
            the event time index before `t_stop`.
            
            Laplacians are computed from `self.times[self._k_start_laplacians]`
            until `self.times[self._k_stop_laplacians-1]`.
            
            The laplacian at step k, is the random walk laplacian
            between `times[k]`. and `times[k+1]`.
            
        Parameters
        ----------
        t_start : float or int, optional
            The default is None, i.e. starts at the beginning of times.
            The computation starts from the first time index before or equal to t_start.
            The corresponding starting time index is saved in `self._k_start_laplacians`
            and the real starting time is `self.times[self._k_start_laplacians]` which 
            is saved in `self._t_start_laplacians`.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations. Default is end of times.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Similarily to `t_start`, the corresponding times are saved in 
            `self._k_stop_laplacians` and `self._t_stop_laplacians`.
        verbose : bool, optional
            The default is False.
        save_adjacencies : bool, optional
            Default is False. Use to save adjacency matrices in `self.adjacencies`.

        Returns
        -------
        None.            
        """
            
        if verbose:
            print('PID ', os.getpid(), ' : ','Computing Laplacians')       
    
        t0 = time.time()
        np_adjacency = self.adjacency.toarray()
        if random_walk is True:
            # Calculate the degree vector (sum of non-zero elements in each row)
            degree_vector = np.array(np_adjacency.sum(axis=1)).ravel()

            # Avoid division by zero by setting degree to 1 for nodes with no edges
            degree_vector[degree_vector == 0] = 1.0

            # Create a diagonal matrix from the inverted degree vector
            degree_matrix = csr_matrix(np.diag(1.0 / degree_vector))

            # Calculate the random walk Laplacian matrix: L_rw = I - D^(-1) * A
            self.laplacians = csr_matrix(np.identity(np_adjacency.shape[0]) - degree_matrix @ np_adjacency)
        else:
            self.laplacians = csr_matrix(diags(np.asarray(np_adjacency.sum(1)).squeeze()) - np_adjacency)                            

        t_end = time.time()-t0
        
        if verbose:
            print('PID ', os.getpid(), ' : ','finished in ', t_end)
       
    def compute_inter_transition_matrices(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    fix_tau_k=False,
                                    use_sparse_stoch=False,
                                    dense_expm=True):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'inter_T'):
            self.inter_T = dict()            
    
        
        if lamda is None:
            if verbose:
                print('PID ', os.getpid(), ' : ','taking lamda as 1/tau_w with tau_w = median interevent time')
            lamda = 1/np.median(np.diff(self.times))
        
        # new value of lamda, we need to recompute
        if lamda not in self.inter_T.keys():
            
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing interevent transition matrices for lamda={0}'.format(lamda))

            self.inter_T[lamda] = []
                
            t0 = time.time()
            tk=self.times[0]
            for k in range(1,len(self.times)):
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' ,
                          self._k_stop_laplacians-1-self._k_start_laplacians)
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                if fix_tau_k:
                    tau_k = 1.0
                else:
                    tau_k = self.times[k] - tk
                
                if use_sparse_stoch:
                    # self.inter_T[lamda].append(sparse_lapl_expm(self.laplacians,
                    #                                                    tau_k*lamda,
                    #                                                    dense_expm=dense_expm))
                    raise Exception("Shouldn't use use_sparse_stoch=True")
                else:
                    if self.laplacians.getnnz() == 0:
                        # expm of zero = identity
                        print('special case: network is empty')
                        self.inter_T[lamda].append(eye(self.num_nodes,format='csr'))
                    else:
                        if dense_expm:
                            #self.inter_T[lamda].append(csr_matrix(expm(-tau_k*lamda*self.laplacians.toarray())))
                            raise Exception("Shouldn't use dense_expm=True")
                        else:
                            self.inter_T[lamda].append(expm(-tau_k*lamda*self.laplacians).tocsr())
                    tk = self.times[k]
        
            if len(self.inter_T[lamda]) == 0:
                if verbose:
                    print('PID ', os.getpid(), ' no events, trans. matrix = identity')
                # is there was no event, the transition is identity
                if use_sparse_stoch:
                    #self.inter_T[lamda].append(sparse_stoch_mat.create_diag(size=self.num_nodes))
                    raise Exception("Shouldn't use use_sparse_stoch=True")
                else:
                    print('Special case, inter T = eye')
                    self.inter_T[lamda].append(eye(self.num_nodes, 
                                                      dtype=np.float64,
                                                      format='csr'))
                
            t_end = time.time()-t0
            tk= self.times[k]
            
            if verbose:
                print('PID ', os.getpid(), ' : ','Finished computing interevent transition matrices in ', t_end)
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Interevent transition matrices already computed for lamda={0}'.format(lamda))

    def compute_transition_matrices(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=False,
                                    tol=None):
        """ compute transition matrices and saves them in a dict of lists `self.T[lamda]` 
            where `self.T[lamda][k]` is the product of all interevent transition 
            matrices from t_0 to t_k computed with lamda.
        """
            
        if not hasattr(self, 'inter_T') or \
                lamda not in self.inter_T.keys():
            raise Exception("Compute inter_T first.")

            
        if not hasattr(self, 'T'):
            self.T = dict()
            
        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))
            
            
        if lamda not in self.T.keys():
            if save_intermediate:
                if force_csr:
                    # forcing the first matrix to csr, will ensure that 
                    # all products are done in csr format,
                    # since CSR @ SparseStochMat t is not implemented
                    self.T[lamda] = [self.inter_T[lamda][k_init].tocsr()]
                else:
                    self.T[lamda] = [self.inter_T[lamda][k_init]]
                
                if tol is not None:
                    set_to_zeroes(self.T[lamda][0],tol)
                    inplace_csr_row_normalize(self.T[lamda][0])
            else:
                if force_csr:
                    self.T[lamda] = self.inter_T[lamda][k_init].tocsr()
                else:
                    self.T[lamda] = self.inter_T[lamda][k_init]
                    
                if tol is not None:
                    set_to_zeroes(self.T[lamda],tol)
                    inplace_csr_row_normalize(self.T[lamda])
                    
                    
            
        
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing transition matrix')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                Tk = self.inter_T[lamda][k]
                if tol is not None:
                    set_to_zeroes(Tk,tol)
                    inplace_csr_row_normalize(Tk)
                    
                if save_intermediate:
                    
                    self.T[lamda].append(self.T[lamda][-1] @ Tk)
                    
                    #normalize T to correct precision errors
                    if tol is not None:
                        set_to_zeroes(self.T[lamda][-1],tol)
                        
                    inplace_csr_row_normalize(self.T[lamda][-1])
                else:
                    self.T[lamda] = self.T[lamda] @ Tk
                    if tol is not None:
                        set_to_zeroes(self.T[lamda], tol)
                        
                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T[lamda])
                
            t_end = time.time()-t0
            
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Transition matrices already computed for lamda={0}'.format(lamda))      

    def compute_entropy(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    tol=None,
                                    time_domain = None):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'S'):
            self.S = dict()


        if time_domain is None:
            time_domain = self.times()
        if reverse_time:
            k_init = len(time_domain)-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(time_domain))
            
        p0 = 1/self.num_nodes*np.ones(self.num_nodes)
        
        if lamda not in self.S.keys():
            #if save_intermediate:
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                p = p0 @ self.T[lamda][k_init].tocsr()
                T = self.T[lamda][k_init].tocsr()
                logTdata = np.log(np.where(T.data > 0, T.data, 1))
                TlogT = csr_matrix((T.data*logTdata, T.indices, T.indptr), shape=T.shape)
                # there shouldn't be need for this
                # TlogT[TlogT>0]=0
                self.S[lamda] = [-np.sum(p0 @ TlogT, where= np.isfinite(p0 @ TlogT))]
            else:
                raise Exception("Use force_csr=True")
            
            if tol is not None:
                set_to_zeroes(self.S[lamda][0],tol)
                # inplace_csr_row_normalize(self.T[lamda][0])
            '''else:
                if force_csr:
                    self.T[lamda] = self.inter_T[lamda][k_init].tocsr()
                else:
                    self.T[lamda] = self.inter_T[lamda][k_init]
                    
                if tol is not None:
                    set_to_zeroes(self.T[lamda],tol)
                    inplace_csr_row_normalize(self.T[lamda])'''
                    
                    
            
        
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                p = p0 @ self.T[lamda][k]
                T = self.T[lamda][k].tocsr()
                logTdata = np.log(np.where(T.data > 0, T.data, 1))
                TlogTdata = T.data * logTdata
                TlogT = csr_matrix((TlogTdata, T.indices, T.indptr), shape=T.shape)
                self.S[lamda].append(-np.sum(p0 @ TlogT, where= np.isfinite(p0 @ TlogT)))

            t_end = time.time()-t0
            
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda)) 

    def compute_vonNeumann_entropy(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    time_domain=None):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'vNS'):
            self.vNS = dict()

        
        if time_domain is None:
            time_domain = self.times()
        if reverse_time:
            k_init = len(time_domain)-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(time_domain))
            
        
        if lamda not in self.vNS.keys():
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                T = self.T[lamda][k_init].tocsr()
                T = T.toarray()
                rho = T/np.trace(T)
                rhologrho = rho @ d_logm(rho)
                self.vNS[lamda] = [- 1/np.log(self.num_nodes) * np.trace(rhologrho)]
                del T
                del rho
                del rhologrho
            else:
                raise Exception
         
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                T = self.T[lamda][k].tocsr()
                T = T.toarray()
                rho = T/np.trace(T)
                rhologrho = rho @ d_logm(rho)
                self.vNS[lamda].append(- 1/np.log(self.num_nodes) * np.trace(rhologrho))
                del T
                del rho
                del rhologrho
            t_end = time.time()-t0
            
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda)) 

    def compute_simple_vonNeumann_entropy(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    time_domain=None):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'simple_vNS'):
            self.simple_vNS = dict()

        
        if time_domain is None:
            time_domain = self.times()
        if reverse_time:
            k_init = len(time_domain)-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(time_domain))
            
        
        if lamda not in self.simple_vNS.keys():
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                T = self.T[lamda][k_init].tocsr()
                T = T.toarray()
                rho = T/np.trace(T)
                # np_rho = rho.toarray()
                # np_logrho = np.log(np.where(np_rho != 0, np_rho, 1))
                # np_rhologrho = np_rho  * np_logrho
                # # np_rhologrho[np.isnan(np_rhologrho)] = 0
                # rhologrho = csr_matrix(np_rhologrho)
                # del np_rhologrho
                # del np_rho
                # del np_logrho
                # self.simple_vNS[lamda] = [-rhologrho.trace()]
                
                L = self.laplacians.toarray()
                self.simple_vNS[lamda] = [1/np.log(self.num_nodes) * (np.trace(self.times[k_init] * rho @ L) + np.log(np.trace(T)))]
            else:
                raise Exception("Use force_csr=True")
                '''p = p0 @ self.T[lamda][k_init]
                logT = np.log(self.T[lamda][k_init])
                TlogT = np.multiply(self.T[lamda][k_init], logT)
                self.rho[lamda] = [np.sum(p @ TlogT)]'''
         
        
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                T = self.T[lamda][k].tocsr()
                T = T.toarray()
                rho = T/np.trace(T)
                # np_rho = rho.toarray()
                # np_logrho = np.log(np.where(np_rho != 0, np_rho, 1))
                # np_rhologrho = np_rho  * np_logrho
                # # np_rhologrho[np.isnan(np_rhologrho)] = 0
                # rhologrho = csr_matrix(np_rhologrho)
                # del np_rhologrho
                # del np_rho
                # del np_logrho
                # self.simple_vNS[lamda].append(-rhologrho.trace())
                L = self.laplacians.toarray()
                self.simple_vNS[lamda].append(1/np.log(self.num_nodes) * np.trace(self.times[k] * rho @ L) + np.log(np.trace(T)))
                del T
                del rho

            t_end = time.time()-t0
            
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda))

    def compute_entropy2(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'S2'):
            self.S2 = dict()

        
        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))
            
        p0 = 1/self.num_nodes*np.ones(self.num_nodes)
        
        if lamda not in self.S2.keys():
            #if save_intermediate:
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                p = p0 @ self.T[lamda][k_init].tocsr()
                diagp = csr_matrix(np.diag(p))
                T = self.T[lamda][k_init].tocsr()
                Q = -lamda*self.laplacians[k_init]
                diagpQ= diagp @ Q
                array_diagpQ = np.asarray(diagpQ.toarray())
                
                entropy2 = 0
                for i in range(array_diagpQ.shape[0]):
                    for j in range(diagpQ.shape[1]):
                        if np.isfinite(np.log(array_diagpQ[i][j]/array_diagpQ[j][i])):
                            entropy2 += array_diagpQ[i][j] * np.log(array_diagpQ[i][j]/array_diagpQ[j][i])
                self.S2[lamda] = [entropy2]
            else:
                raise Exception("Use force_csr=True")
  
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy2 ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                p = p0 @ self.T[lamda][k].tocsr()
                diagp = csr_matrix(np.diag(p))
                T = self.T[lamda][k].tocsr()
                Q = -lamda*self.laplacians[k]
                diagpQ= diagp @ Q
                array_diagpQ = np.asarray(diagpQ.toarray())
                
                entropy2 = 0
                for i in range(array_diagpQ.shape[0]):
                    for j in range(diagpQ.shape[1]):
                        if np.isfinite(np.log(array_diagpQ[i][j]/array_diagpQ[j][i])):
                            entropy2 += array_diagpQ[i][j] * np.log(array_diagpQ[i][j]/array_diagpQ[j][i])
                self.S2[lamda].append(entropy2)

                
            t_end = time.time()-t0
            
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy2 already computed for lamda={0}'.format(lamda)) 

    def compute_new_vonNeumann_entropy(self, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    time_domain=None):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   
    
        if not hasattr(self, 'new_vNS'):
            self.new_vNS = dict()

        
        if time_domain is None:
            time_domain = self.times()
        if reverse_time:
            k_init = len(time_domain)-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = len(time_domain)-1
            k_range = range(1,len(time_domain))
            
        
        if lamda not in self.new_vNS.keys():
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                L = self.laplacians
                T = self.T[lamda][k_init].tocsr()
                rho = T/T.trace()
                t0= time_domain[0]
                self.new_vNS[lamda] = [- lamda * t0 * (rho @ L).trace() + np.log(T.trace())]
            else:
                raise Exception
         
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                L = self.laplacians
                T = self.T[lamda][k].tocsr()
                rho = T/T.trace()
                tk= time_domain[k]
                self.new_vNS[lamda].append(- lamda * tk * (rho @ L).trace() + np.log(T.trace()))
              
            t_end = time.time()-t0
            
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda))
    
    def compute_spectral_vonNeumann_entropy(self, lamda=None, t_start=None, t_stop=None,
                                verbose=False,
                                # save_intermediate=True,
                                reverse_time=False,
                                force_csr=True,
                                time_domain=None):
        """
        
        Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        verbose : bool, optional
            The default is False.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        
        if not hasattr(self, 'laplacians'):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop,
                                            verbose=verbose)
        if not hasattr(self, 'inter_T'):
            self.compute_inter_transition_matrices()

        if not hasattr(self, 'T'):
            self.compute_transition_matrices()   

        if not hasattr(self, 'spectral_vNS'):
            self.spectral_vNS = dict()

        
        if time_domain is None:
            time_domain = self.inter_T[lamda]
        if reverse_time:
            k_init = len(time_domain)-1
            k_range = reversed(range(0, k_init))
            if verbose:
                print('PID ', os.getpid(), ' : reversed time computation.')
        else:
            k_init = 0
            k_range = range(1,len(time_domain))
            
        
        if lamda not in self.spectral_vNS.keys():
            if force_csr:
                # forcing the first matrix to csr, will ensure that 
                # all products are done in csr format,
                # since CSR @ SparseStochMat t is not implemented
                T = self.T[lamda][k_init].tocsr()
                T = T.toarray()
                rho = T/T.trace()
                # spectrum_rho = [z.real for z in eigsh(rho, k=self.num_nodes-2)[0]] # num_nodes -2 becusae of limits of eigs in number of eigevanlues
                # spectrum_rho = np.array(spectrum_rho)
                # log_spectrum_rho = np.log(np.where(spectrum_rho > 0, spectrum_rho, 1)) # make sure to have only positive eigenvalues
                # spectrum_vNS = - np.sum(spectrum_rho*log_spectrum_rho) / (self.num_nodes-2)
                # self.spectral_vNS[lamda] = [spectrum_vNS]

                eigs = np.linalg.eigvalsh(rho) # this is for numpy arrays and it returns all the eigenvalues
                eigs = np.where(eigs > 0, eigs, 1)
                self.spectral_vNS[lamda] = [- 1/np.log(self.num_nodes) * np.sum(eigs*np.log(eigs))]
                del T
                del rho
            else:
                raise Exception
            
            if verbose:
                print('PID ', os.getpid(), ' : ','Computing entropy ')
                    
            t0 = time.time()
    
            for k in k_range:
                if verbose and not k%1000:
                    print('PID ', os.getpid(), ' : ',k, ' over ' , len(self.inter_T[lamda]))
                    print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
                    
                T = self.T[lamda][k].tocsr()
                T = T.toarray()
                rho = T/T.trace()
                # spectrum_rho = [z.real for z in eigsh(rho, k=self.num_nodes-2)[0]] # num_nodes -2 becusae of limits of eigs in number of eigevanlues
                # spectrum_rho = np.array(spectrum_rho)
                # log_spectrum_rho = np.log(np.where(spectrum_rho > 0, spectrum_rho, 1)) # make sure to have only positive eigenvalues
                # spectrum_vNS = - np.sum(spectrum_rho*log_spectrum_rho) / (self.num_nodes-2)
                # self.spectral_vNS[lamda].append(spectrum_vNS)

                eigs = np.linalg.eigvalsh(rho) # this is for numpy arrays and it returns all the eigenvalues
                eigs = np.where(eigs > 0, eigs, 1)
                self.spectral_vNS[lamda].append(- 1/np.log(self.num_nodes) * np.sum(eigs*np.log(eigs)))
                del T
                del rho
            
            t_end = time.time()-t0
            
            
            if verbose:
                print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s')
        else:
            if verbose:
                print('PID ', os.getpid(), ' : ','Entropy already computed for lamda={0}'.format(lamda)) 