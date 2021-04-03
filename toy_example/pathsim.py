import networkx as nx
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from collections import Counter
from operator import itemgetter
import copy

class PathSimRec:

    def __init__(self, N=1):
        self.N = N


    def compute_similarity_matrix(self, G, target_entity, adj_entity, path_len):
        """
        Parameters
        -----------
        G : graph
        target_entity : the entity, for the nodes of which similarities should be calculated
        adj_entity : the entity, from the node of which the paths start
        path_len : the length of the path between target and adj nodes

        Returns
        -------
        sim : similarity matrix for target entity nodes
        """

        self.G = G

        target_nodes = [node for node, attr in G.nodes(data=True) if attr['entity'] == target_entity]
        adj_nodes = [node for node, attr in G.nodes(data=True) if attr['entity'] == adj_entity]

        # ------ Calculating Adjacency matrix between target and adjacency nodes
        adj = defaultdict(list)

        for n1 in target_nodes:
            adj[n1] = defaultdict(list)
            for n2 in adj_nodes:
                adj[n1][n2] = len([path for path in nx.all_simple_paths(G, n1, n2, path_len)])

        # ------ Calculating similarities between target nodes
        sim = defaultdict(list)
        n_paths = defaultdict(list)

        for n1 in target_nodes:
            sim[n1] = defaultdict(int) #list
            n_paths[n1] = defaultdict(int)
            for n2 in target_nodes:
                sim[n1][n2] = -1
                n_paths[n1][n2] = 0

        for n1 in target_nodes:
            for n2 in target_nodes:
                if sim[n2][n1] == -1:
                    n_connecting_paths = 0
                    numerator = 0
                    denominator = 0
                    for n3 in adj_nodes:
                        n_connecting_paths += adj[n1][n3] * adj[n2][n3]
                        numerator += 2 * adj[n1][n3] * adj[n2][n3]
                        denominator += adj[n1][n3] ** 2 + adj[n2][n3] ** 2
                    sim[n1][n2] = numerator / denominator if denominator != 0 else 0
                    sim[n2][n1] = sim[n1][n2]
                    n_paths[n1][n2] = n_connecting_paths
                    n_paths[n2][n1] = n_paths[n1][n2]

        self.itemitem_matrix = sim
        self.n_paths_matrix = n_paths


    def compute_similarity_matrix_my(self, G, target_entity, adj_entity, path_len):
        """
        Parameters
        -----------
        G : graph
        target_entity : the entity, for the nodes of which similarities should be calculated
        adj_entity : the entity, from the node of which the paths start
        path_len : the length of the path between target and adj nodes

        Returns
        -------
        sim : similarity matrix for target entity nodes
        """

        self.G = G

        target_nodes = [node for node, attr in G.nodes(data=True) if attr['entity'] == target_entity]
        adj_nodes = [node for node, attr in G.nodes(data=True) if attr['entity'] == adj_entity]

        # ------ Calculating Adjacency matrix between target and adjacency nodes
        adj = defaultdict(list)

        for n1 in target_nodes:
            adj[n1] = defaultdict(list)
            for n2 in adj_nodes:
                adj[n1][n2] = len([path for path in nx.all_simple_paths(G, n1, n2, path_len)])

        # ------ Calculating similarities between target nodes
        sim = defaultdict(list)
        for i, n1 in enumerate(target_nodes):
            sim[n1] = defaultdict(list)
            for j, n2 in enumerate(target_nodes):
                sim[n1][n2] = 0

        for i, n1 in enumerate(target_nodes):
            for j, n2 in enumerate(target_nodes):
                if i < j:
                    n_connecting_paths = 0
                    for n3 in adj_nodes:
                        n_connecting_paths += adj[n1][n3] * adj[n2][n3]
                    sim[n1][n2] = n_connecting_paths
                    sim[n2][n1] = sim[n1][n2]

        # Normalization

        sim_norm = copy.deepcopy(sim)
        for i, n1 in enumerate(target_nodes):
            row_sum = sum(sim[n1].values())
            for j, n2 in enumerate(target_nodes):
                sim_norm[n1][n2] = sim[n1][n2] / row_sum if row_sum != 0 else 0

        self.itemitem_matrix = sim_norm


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

        return user_rec[:self.N]


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
            sessions = [(k,v) for k, v in sorted(session_relevance_vector.items(), key=itemgetter(1), reverse=True) if k != session and v != 0]
            if len(sessions) == 0:
                return []

            k_sessions = []
            sessionitem_matrix = defaultdict(list)
            for s, score in sessions:
                articles = [a for a in self.G[s] if self.G[s][a]['edge_type'] == type and a not in item_list]
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
                        k != session and v != 0][:kNN]

        return sessions

    def get_similar_sessions(self, session, threshold=0):

        session_relevance_vector = self.itemitem_matrix[session]

        if len(session_relevance_vector) > 0:
            sessions = [k for k, v in sorted(session_relevance_vector.items(), key=itemgetter(1), reverse=True)
                        if k != session and v > threshold]

        return sessions


    def predict_next_by_SB(self, session, item_list, topN=True,type='SA'):

        similar_sessions = self.get_similar_sessions(session)

        articles = []
        for s in similar_sessions:
            articles.extend([a for a in self.G[s] if self.G[s][a]['edge_type'] == type and a not in item_list])

        articles_freq = Counter(articles)

        if topN == True:
            rec = dict(articles_freq.most_common(self.N))
        else:
            rec = dict(articles_freq.most_common())

        return rec


    def get_avg_n_of_connected_sessions(self):

        connected_sessions = dict()
        for s in self.n_paths_matrix:
            # print(self.n_paths_matrix[s])
            if len(self.n_paths_matrix[s])==0:
                connected_sessions[s] = 0
            else:
                connected_sessions[s] = sum([1 if v > 0 else 0 for v in self.n_paths_matrix[s].values()])
        if len(connected_sessions)==0:
            avg_n_of_connected_sessions = 0
        else:
            # avg_n_of_connected_sessions = float(sum(connected_sessions.values())) / len(connected_sessions)
            #
            avg_n_of_connected_sessions = float(sum(connected_sessions.values()))
#         print(f'Connected Sessions : {connected_sessions}')
        return avg_n_of_connected_sessions

    # def get_avg_n_of_connected_sessions(self):
    #
    #     #print('n_paths_matrix:\n', self.n_paths_matrix)
    #
    #     connected_sessions = dict()
    #     for s in self.n_paths_matrix:
    #         if len(self.n_paths_matrix[s]) > 0:
    #             return 1
    #         n_connected_sessions = sum([1 if v > 0 else 0 for v in self.n_paths_matrix[s].values()])
    #         if n_connected_sessions != 0:
    #             connected_sessions[s] = n_connected_sessions
    #
    #     avg_n_of_connected_sessions = float(sum(connected_sessions.values())) / len(connected_sessions)
    #
    #     return avg_n_of_connected_sessions

    def get_avg_n_of_connected_articles(self):

        # print('n_paths_matrix:\n', pd.DataFrame(self.n_paths_matrix))

        connected_articles = dict()
        for s in self.n_paths_matrix:
            n_connected_articles = sum([1 if v > 0 else 0 for v in self.n_paths_matrix[s].values()])
            if n_connected_articles != 0:
                connected_articles[s] = n_connected_articles
        if len(connected_articles)==0:
            avg_n_of_connected_articles = 0
        else:
            # avg_n_of_connected_articles = float(sum(connected_articles.values())) / len(connected_articles)
            avg_n_of_connected_articles = float(sum(connected_articles.values()))
        return avg_n_of_connected_articles



    def predict_next_by_AB(self, articles, option='ib', topN = True):

        '''
        option = ['ib', 'sb'] = [item based, session based]
        item based - only the last item matters
        session based - all previously read articles in the session matter
        '''

        if option == 'ib':
            a_len = len(articles)
            if a_len > 1:
                a = articles[a_len-1]
                item_list = articles[:a_len]
            else:
                a = articles[0]
                item_list = articles

            article_paths_vector = self.n_paths_matrix[a]

            if len(article_paths_vector) > 0:
                connected_articles = [(k, v) for k, v in sorted(article_paths_vector.items(), key=itemgetter(1), reverse=True)
                            if k not in item_list and v > 0]
                if topN == True:
                    connected_articles = connected_articles[:self.N]
            else:
                connected_articles = []

        elif option == 'sb':

            item_list = articles

            paths_list = []
            for i, item in enumerate(item_list):
                if len(self.n_paths_matrix[item]) > 0:
                    item_paths = self.n_paths_matrix[item]
                    paths_list.append(item_paths)
            # print('paths_list:', paths_list)

            if len(paths_list) > 0:
                paths_sum_dict = dict(pd.DataFrame(paths_list).fillna(0).sum())
                # print('paths_sum_dict:', paths_sum_dict)
                connected_articles = [(k, v) for k, v in sorted(paths_sum_dict.items(), key=itemgetter(1), reverse=True)
                                      if k not in item_list and v > 0]
                if topN == True:
                    connected_articles = connected_articles[:self.N]
            else:
                connected_articles = []


        return dict(connected_articles)


    def get_similar_users(self, user, recent_users, threshold = 0):

        user_relevance_vector = self.itemitem_matrix[user]

        if len(user_relevance_vector) > 0:
            users = [k for k, v in sorted(user_relevance_vector.items(), key=itemgetter(1), reverse=True)
                        if k != user and k in recent_users and v > threshold]

        return users



    def predict_next_by_UB(self, similar_users, item_list, g, topN=True,entity = 'A'):

        articles = []
        for u in similar_users:
            articles.extend([a for a, attr in g.nodes(data=True)
                             if attr['entity']==entity
                             and a not in item_list
                             and nx.has_path(g,u,a)
                             and nx.shortest_path_length(g,u,a)==2])

        articles_freq = Counter(articles)

        if topN == True:
            rec = dict(articles_freq.most_common(self.N))
        else:
            rec = dict(articles_freq.most_common())

        return rec