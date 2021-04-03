import networkx as nx
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from operator import itemgetter

class PersonalizedPageRankBasedRec:

    def __init__(self, N=1,custom_user='U',custom_session='S',custom_article='A'):
        self.N = N
        self.custom_user= custom_user
        self.custom_session=custom_session
        self.custom_article=custom_article

    def compute_transition_matrix(self, G, beta=0.8, eps=1e-4, max_iter=100):
        # ----- Calculate transition matrix -----
        self.G = G

        # Basic parameters
        self.n = nx.number_of_nodes(self.G)
        I = np.identity(self.n)

        if self.n == 0:
            self.V = I
            return

        # Map a node to its sequential number
        self.n_dict = dict(zip(self.G.nodes(), range(self.n)))

        # Creating a transition matrix M
        A = nx.adjacency_matrix(self.G)
        if len(self.G.degree) ==1:
            d = list(self.G.degree())[0][1]
            d = np.array(d).reshape(1,1)
        else:
            d = np.array(self.G.degree())[:,1].astype(int)
        #p = 1. / d
        p = np.array(list(map(lambda x: 1. / x if x != 0 else 0, d)))
        M = A.multiply(p)

        # Implementing Random Walk with Restart algorithm
        self.V = I
        for i in range(max_iter):
            V_new = beta * M * self.V + (1 - beta) * I

            if (abs(self.V - V_new) < eps).all():
                #print("Converged after %d iteration" % (i))
                break

            self.V = V_new

    def transform_matrix_to_dataframe(self):

        # ----- Transform a matrix to a dataframe -----
        # COLUMNS OF THE DATAFRAME SUM UP TO 1 (NOT ROWS!)
        # Map a node to its sequential number
        n_dict = dict(zip(self.G.nodes(), range(self.n)))

        # Create a dictionary on the base of the matrix, with assigning node names to the keys
        result = defaultdict(list)
        for n1 in self.G.nodes():
            result[n1] = defaultdict(int)
            for n2 in self.G.nodes():
                result[n1][n2] = self.V[n_dict[n2]][n_dict[n1]]

        # Save the result to the data frame object
        self.V_df = pd.DataFrame(result)#.round(4)

    def create_itemitem_matrix(self):

        item_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == self.custom_article]

        itemitem_matrix = defaultdict(list)
        for n1 in item_nodes:
            itemitem_matrix[n1] = defaultdict(int)
            for n2 in item_nodes:
                itemitem_matrix[n1][n2] = self.V[self.n_dict[n2]][self.n_dict[n1]]

        self.itemitem_matrix = itemitem_matrix

    def create_useritem_matrix(self):

        item_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == self.custom_article]
        user_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == self.custom_user]

        useritem_matrix = defaultdict(list)
        for n1 in user_nodes:
            useritem_matrix[n1] = defaultdict(int)
            for n2 in item_nodes:
                useritem_matrix[n1][n2] = self.V[self.n_dict[n2]][self.n_dict[n1]]

        self.useritem_matrix = useritem_matrix

    def create_usercategory_matrix(self, user_nodes=None):

        category_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == 'C']
        if user_nodes == None:
            user_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == self.custom_user]

        usercategory_matrix = defaultdict(list)
        for n1 in user_nodes:
            usercategory_matrix[n1] = defaultdict(int)
            for n2 in category_nodes:
                usercategory_matrix[n1][n2] = self.V[self.n_dict[n2]][self.n_dict[n1]]

        self.usercategory_matrix = usercategory_matrix

    def create_categorycategory_matrix(self):

        category_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == 'C']

        categorycategory_matrix = defaultdict(list)
        for n1 in category_nodes:
            categorycategory_matrix[n1] = defaultdict(int)
            for n2 in category_nodes:
                categorycategory_matrix[n1][n2] = self.V[self.n_dict[n2]][self.n_dict[n1]]

        self.categorycategory_matrix = categorycategory_matrix

    def create_sessionsession_matrix(self):

        session_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == self.custom_session]

        sessionsession_matrix = defaultdict(list)
        for n1 in session_nodes:
            sessionsession_matrix[n1] = defaultdict(int)
            for n2 in session_nodes:
                sessionsession_matrix[n1][n2] = self.V[self.n_dict[n2]][self.n_dict[n1]]

        self.sessionsession_matrix = sessionsession_matrix

    def create_sessionitem_matrix(self):

        session_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == self.custom_session]
        item_nodes = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == self.custom_article]

        sessionitem_matrix = defaultdict(list)
        for n1 in session_nodes:
            sessionitem_matrix[n1] = defaultdict(int)
            for n2 in item_nodes:
                sessionitem_matrix[n1][n2] = self.V[self.n_dict[n2]][self.n_dict[n1]]

        self.sessionitem_matrix = sessionitem_matrix



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

    @staticmethod
    def assign_timeviews_weights(t_list):

        w_list = [np.log(t+1) if t > 20 else np.log10(t+1) for t in t_list]

        # Normalize
        s = sum(w_list)
        w_list = [w / s for w in w_list]

        return w_list


    def predict_next(self, user, item_list, method=1, timeviews=None, order_vec=None):
        '''
        Given user and the list of already viewed items in the test session predict a next item
        returning a ranked list of N predictions

        method:
        1 - normal mean,
        2 - weighted mean (sigmoid)
        3 - weighted mean (timeviews)
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
        # return user_rec[:self.N]



    def predict_next_ic(self, item_rel_vector, item_cat_vector, cat_rel_vector=None, user_cat_rel_vector=None, w=None):



        if cat_rel_vector != None and user_cat_rel_vector == None: # SM
            cat_relevance_vector = [cat_rel_vector[cat]
                                    if cat in cat_rel_vector else 0
                                    for cat in item_cat_vector]
            # mult = [i_rel * c_rel for i_rel, c_rel in zip(item_rel_vector.values(), cat_relevance_vector)]
            mult = [0.7*i_rel + 0.3*c_rel for i_rel, c_rel in zip(item_rel_vector.values(), cat_relevance_vector)]
        elif cat_rel_vector == None and user_cat_rel_vector != None: # SL
            user_cat_rel_vector = [user_cat_rel_vector[cat]
                                   if cat in user_cat_rel_vector else 0
                                   for cat in item_cat_vector]
            mult = [0.7 * i_rel + 0.3 * uc_rel for i_rel, uc_rel in zip(item_rel_vector.values(), user_cat_rel_vector)]
        else:
            cat_relevance_vector = [cat_rel_vector[cat]
                                    if cat in cat_rel_vector else 0
                                    for cat in item_cat_vector]
            user_cat_rel_vector = [user_cat_rel_vector[cat]
                                   if cat in user_cat_rel_vector else 0
                                   for cat in item_cat_vector]
            # mult = [i_rel * c_rel * uc_rel for i_rel, c_rel, uc_rel in zip(item_rel_vector.values(), cat_relevance_vector, user_cat_rel_vector)]
            mult = [0.7*i_rel + 0.2*c_rel + 0.1*uc_rel for i_rel, c_rel, uc_rel in
                    zip(item_rel_vector.values(), cat_relevance_vector, user_cat_rel_vector)]

        final_dict = {key: val for key, val in zip(item_rel_vector.keys(), mult)}
        rec = [k for k, v in sorted(final_dict.items(), key=itemgetter(1), reverse=True)][:self.N]

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
        t_list = []
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
                    t_list.append(timeviews[i])
                    # w_t_list.append(np.log(timeviews[i]))

        if method == 3:
            s = sum(w_t_list)
            if s != 0:
                w_t_list = [w/s for w in w_t_list]
            else:
                w_t_list = [1 if i == len(w_t_list)-1 else 0 for i,w in enumerate(w_t_list)]


        if len(sim_list) > 0:
            # --- Create a dict of weighted scores for each article in the list
            if method == 0:
                sim_mean_dict = dict(self.itemitem_matrix[sim_list[-1]])
            elif method == 1:
                sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).mean())
            elif method == 2:
                w_s_list = self.assign_sigmoid_weights(helpful_item_list)
                sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).multiply(w_s_list, axis='rows').sum())
            elif method == 3:
                w_t_list = self.assign_timeviews_weights(t_list)
                sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).multiply(w_t_list, axis='rows').sum())

            # Normalize
            sum_dict = sum(sim_mean_dict.values())
            norm_sim_mean_dict = {key: value / sum_dict if sum_dict != 0 else 0 for key, value in sim_mean_dict.items()}

        else:
            norm_sim_mean_dict = []

        return norm_sim_mean_dict



    def get_category_relevance_vector(self, category_list, method=1, timeviews=None):
        '''
        method 1 - normal mean,
        method 2 - sigmoid f-n,
        method 3 - timeviews
        '''

        cat_trans_list = []
        w_t_list = []
        t_list = []
        for i, cat in enumerate(category_list):
            if cat in self.categorycategory_matrix and len(self.categorycategory_matrix[cat]) > 0:
                cat_trans_list.append(self.categorycategory_matrix[cat])
                if method == 3:
                    w_t_list.append(np.log(timeviews[i] + 1))
                    t_list.append(timeviews[i])
            else:
                category_list.remove(cat)

        # print('cat_trans_list:', cat_trans_list)

        if len(cat_trans_list) > 0:
            # --- Create a dict of weighted scores for each category in the list
            if method == 0:
                sim_mean_dict = dict(self.itemitem_matrix[cat_trans_list[-1]])
            elif method == 1:
                trans_mean_dict = dict(pd.DataFrame(cat_trans_list).fillna(0).mean())
            elif method == 2:
                w_s_list = self.assign_sigmoid_weights(cat_trans_list)
                trans_mean_dict = dict(pd.DataFrame(cat_trans_list).fillna(0).multiply(w_s_list, axis='rows').sum())
            elif method == 3:
                # timeviews = [t-1 for t in timeviews]
                w_t_list = self.assign_timeviews_weights(t_list)
                trans_mean_dict = dict(pd.DataFrame(cat_trans_list).fillna(0).multiply(w_t_list, axis='rows').sum())

            # print('trans_mean_dict:', trans_mean_dict)

            # Normalize
            sum_dict = sum(trans_mean_dict.values())
            norm_trans_mean_dict = {key: value / sum_dict for key, value in trans_mean_dict.items()}

            # print('trans_mean_dict:', trans_mean_dict)
            # print('norm_trans_mean_dict:', norm_trans_mean_dict)

        else:
            norm_trans_mean_dict = []

        return norm_trans_mean_dict



    def get_user_cat_relevance_vector(self, user, category_list, method=1, timeviews=None):
        '''
        method 1 - normal mean,
        method 2 - sigmoid f-n,
        method 3 - timeviews
        '''

        cat_trans_list = []
        w_t_list = []
        t_list = []
        for i, cat in enumerate(category_list):
            cat_trans_list.append(self.usercategory_matrix[user][cat])
            if method == 3:
                w_t_list.append(np.log(timeviews[i]+1))
                t_list.append(timeviews[i])

        # print('cat_trans_list:', cat_trans_list)

        if len(cat_trans_list) > 0:
            # --- Create a dict of weighted scores for each category in the list
            if method == 1:
                trans_mean_dict = cat_trans_list
            elif method == 2:
                w_s_list = self.assign_sigmoid_weights(category_list)
                trans_mean_dict = [value * weight for value, weight in zip(cat_trans_list, w_s_list)]
            elif method == 3:
                w_t_list = self.assign_timeviews_weights(t_list)
                trans_mean_dict = [value * weight for value, weight in zip(cat_trans_list, w_t_list)]

            # print('trans_mean_dict:', trans_mean_dict)

            # Normalize
            sum_dict = sum(trans_mean_dict)
            norm_trans_mean_dict = {key: value / sum_dict for key, value in zip(category_list, trans_mean_dict)}

            # print('norm_trans_mean_dict:', norm_trans_mean_dict)

        else:
            norm_trans_mean_dict = []

        return norm_trans_mean_dict


    # def predict_next_using_categories(self, user, item_list, cc_matrix, full_G):
    #
    #     sim_list = []
    #
    #     no_rec = True
    #     for item in item_list:
    #         if len(self.itemitem_matrix[item]) > 0:
    #             sim_list.append(self.itemitem_matrix[item])
    #             sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).mean())
    #             no_rec = False
    #
    #     def map_category(G, a):
    #         c = [c for c in self.G[a] if self.G[a][c]['edge_type'] == 'AC']
    #         c = c[0] if len(c) > 0 else ''
    #         return c
    #
    #     if no_rec == False:
    #         # Create category vector
    #         last_item = item_list[len(item_list)]
    #         target_cat = map_category(G, last_item)
    #         #.....
    #         user_rec = sorted(sim_mean_dict, key=lambda x: x[1], reverse=True)[:self.N]
    #     else:
    #         user_rec = []
    #
    #     return user_rec



    def predict_next_by_sessionKNN(self, session, kNN, method=2):
        '''
        Given current session predict articles by finding the most similar sessions to the current one
        '''

        session_relevance_vector = self.sessionsession_matrix[session]

        if len(session_relevance_vector) > 0:
            k_sessions = [(k,v) for k, v in sorted(session_relevance_vector.items(), key=itemgetter(1), reverse=True) if v != 0][:kNN]
            if len(k_sessions) == 0:
                return []
            k_sim_sessions = [k for k,v in k_sessions]
            k_session_scores = [v for k,v in k_sessions]

            sim_list = []
            for session in k_sim_sessions:
                sim_list.append(self.sessionitem_matrix[session])


            if len(sim_list) > 0:
                # --- Create a dict of weighted scores for each article in the list
                if method == 1:
                    sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).mean())
                elif method == 2:
                    sim_mean_dict = dict(pd.DataFrame(sim_list).fillna(0).multiply(k_session_scores, axis='rows').sum())

            # Normalize
            sum_dict = sum(sim_mean_dict.values())
            norm_sim_mean_dict = {key: value / sum_dict for key, value in sim_mean_dict.items()}

            if len(norm_sim_mean_dict) > 0:
                user_rec = [k for k, v in sorted(norm_sim_mean_dict.items(), key=itemgetter(1), reverse=True)]
            else:
                user_rec = []
        else:
            user_rec = []

        # print([(k,v) for k, v in sorted(norm_sim_mean_dict.items(), key=itemgetter(1), reverse=True)][:self.N])
        rec = {}
        for i in user_rec[:self.N]:
            rec[i] = 1
        # return user_rec[:self.N]
        return rec


