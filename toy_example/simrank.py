import networkx as nx
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from collections import Counter
from operator import itemgetter
import copy

class SimRankRec:

    def __init__(self, N=1,custom_user='U',custom_session='S',custom_article='A'):
        self.N = N
        self.custom_user = custom_user
        self.custom_session = custom_session
        self.custom_article = custom_article

    # ----------------------------------------------------------------------------------------------
    # ------------------ CHECK FOR CONVERGENCE -----------------------------------------------------
    @staticmethod
    def _is_converge(s1, s2, eps=1e-4):
        vec = []
        for i in s1.keys():
            for j in s1[i].keys():
                vec.append(abs(s1[i][j] - s2[i][j]) < eps)
        return all(vec)

    # ----------------------------------------------------------------------------------------------
    # ------------------------- SIMRANK ------------------------------------------------------------

    def compute_similarity_matrix(self, G, c=0.8, max_iter=100, result_type='A'):
        """Return the SimRank similarity between nodes

        Parameters
        -----------
        G : graph : An UNDIRECTED NetworkX (multi)graph
        c : float, 0 < c <= 1 : Dumping factor (relative importance between in-direct neighbors and direct neighbors)
        max_iter : integer : Maximum number of iterations for simrank calculation
        dump_process : boolean : If true, the calculation process is dumped
        result_type : character : Which entity-entity similarity matrix to return

        Returns
        -------
        simrank: dictionary of dictionary of double : entity-entity similarity matrix
        """

        # --- Initialization

        sim_old = defaultdict(list)
        sim = defaultdict(list)
        degree = defaultdict(list)

        for n in G.nodes():
            sim[n] = defaultdict(int)
            sim[n][n] = 1
            sim_old[n] = defaultdict(int)
            sim_old[n][n] = 0
            degree[n] = G.degree(n)  # how many edges the node has

        # --- Calculate simrank

        # We don't consider in-bound or out-bound links here, only a fact of being connected matters,
        # because it comes naturally, that users are only connected with items (~outbound) and items with users (~inbound)

        for iter_ctr in range(max_iter):

            if self._is_converge(sim, sim_old):
                #print("Converged after %d iteration" % (iter_ctr - 1))
                break

            sim_old = copy.deepcopy(sim)

            for i, u in enumerate(G.nodes()):
                for j, v in enumerate(G.nodes()):

                    if u == v:
                        continue

                    if j > i:
                        s_uv = 0.0
                        for n_u in G.neighbors(u):
                            for n_v in G.neighbors(v):
                                # There can be more than one link connecting two nodes (e.g. Author-Venue)
                                s_uv += sim_old[n_u][n_v] * G.number_of_edges(u, n_u) * G.number_of_edges(v, n_v)

                        # Calculate similarity between nodes
                        sim[u][v] = (c * s_uv / (degree[u] * degree[v])) if (degree[u] * degree[v]) > 0 else 0
                        # Matrix is symmetric, no need to go through all nodes twice
                        sim[v][u] = sim[u][v]

        # --- Result
        sim_result = defaultdict(list)
        result_nodes = [n for n, attr in G.nodes(data=True) if attr['entity'] == result_type]

        for n1 in result_nodes:
            sim_result[n1] = defaultdict(int)
            for n2 in result_nodes:
                sim_result[n1][n2] = sim[n1][n2]

        self.itemitem_matrix = sim_result


    @staticmethod
    def assign_sigmoid_weights(a_list):

        n = len(a_list)
        w_list = []
        for i, a in enumerate(a_list):
            # w_list.append(1 / (1 + math.exp(-(i+1))))
            w_list.append(1 / (1 + math.exp(-(-5 + 10 * (i + 1) / n))))

        # Normalize
        s = sum(w_list)
        w_list = [w / s for w in w_list]

        return w_list


    def predict_next(self, user, item_list, method=1, timeviews=None, order_vec=None):
        '''
        Given user and the list of already viewed items in the test session predict a next item
        returning a ranked list of N predictions

        method: 1 - normal mean, 2 - weighted mean (sigmoid)
        '''

        sim_mean_dict = self.get_item_relevance_vector(item_list=item_list, method=method, timeviews=timeviews)

        if len(sim_mean_dict) > 0:
            user_rec = [k for k, v in sorted(sim_mean_dict.items(), key=itemgetter(1), reverse=True)]
            if order_vec != None:
                base_vec = user_rec
                user_rec = []
                for item in order_vec:
                    if item in base_vec:
                        user_rec.append(item)
        else:
            user_rec = []
        rec = {}
        for i in user_rec[:self.N]:
            rec[i] = 1
        # return user_rec[:self.N]
        return rec

    def get_item_relevance_vector(self, item_list, method=1, timeviews=None):
        '''
        method 1 - normal mean,
        method 2 - sigmoid f-n
        method 3 - timeviews
        '''

        sim_list = []
        helpful_item_list = []
        w_t_list = []
        for i, item in enumerate(item_list):
            if len(self.itemitem_matrix[item]) > 0:
                item_sim = self.itemitem_matrix[item]

                for key in item_list:
                    if key in item_sim:
                        del item_sim[key]

                sim_list.append(item_sim)
                helpful_item_list.append(item)
                if method == 3:
                    w_t_list.append(np.log(timeviews[i] + 1))


        if len(sim_list) > 0:
            # print('sim_list:', sim_list)
            # print('type(sim_list):', type(sim_list))
            # print('sim_list[-1]:', sim_list[-1])
            # --- Create a dict of weighted scores for each article in the list
            if method == 0:
                sim_mean_dict = dict(self.itemitem_matrix[item_list[-1]])
            elif method == 1:
                sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).mean())
            elif method == 2:
                w_s_list = self.assign_sigmoid_weights(helpful_item_list)
                sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).multiply(w_s_list, axis='rows').sum())
            elif method == 3:
                sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).multiply(w_t_list, axis='rows').sum())

            # Normalize
            sum_dict = sum(sim_mean_dict.values())
            norm_sim_mean_dict = {key: value / sum_dict if sum_dict != 0 else 0 for key, value in sim_mean_dict.items()}

        else:
            norm_sim_mean_dict = []

        return norm_sim_mean_dict


    def predict_next_by_sessionKNN(self, session, item_list, kNN):
        '''
        Given currently analysed session find k the most similar ones in the train and recommend
        '''

        session_relevance_vector = self.itemitem_matrix[session]


        if len(session_relevance_vector) > 0:
            sessions = [(k,v) for k, v in sorted(session_relevance_vector.items(), key=itemgetter(1), reverse=True) if k != session]

            k_sessions = []
            sessionitem_matrix = defaultdict(list)
            for s, score in sessions:
                articles = [a for a in self.G[s] if self.G[s][a]['edge_type'] == f'{self.custom_session}{self.custom_article}' and a not in item_list]
                if len(articles) > 0:
                    if not sessionitem_matrix[s]:
                        sessionitem_matrix[s] = defaultdict(int)
                    k_sessions.append((s, self.itemitem_matrix[session][s]))
                    for a in articles:
                        sessionitem_matrix[s][a] = score
                    if len(k_sessions) == kNN:
                        break

            sim_list = []
            for s,_ in k_sessions:
                sim_list.append(sessionitem_matrix[s])

            sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).mean())

            # Normalize
            sum_dict = sum(sim_mean_dict.values())
            norm_sim_mean_dict = {key: value / sum_dict if sum_dict != 0 else 0 for key, value in sim_mean_dict.items()}

            user_rec = [k for k, v in sorted(norm_sim_mean_dict.items(), key=itemgetter(1), reverse=True)][:self.N]

        else:
            user_rec = []

        return user_rec



    def get_k_nearest_sessions(self, session, kNN):

        session_paths_vector = self.n_paths_matrix[session]

        if len(session_paths_vector) > 0:
            sessions = [(k, v) for k, v in sorted(session_paths_vector.items(), key=itemgetter(1), reverse=True) if
                        k != session][:kNN]

        return sessions

    def get_similar_sessions(self, session, threshold=0):

        session_relevance_vector = self.itemitem_matrix[session]

        if len(session_relevance_vector) > 0:
            sessions = [k for k, v in sorted(session_relevance_vector.items(), key=itemgetter(1), reverse=True)
                        if k != session and v > threshold]

        return sessions