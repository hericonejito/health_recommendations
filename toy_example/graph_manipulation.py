import pandas as pd
import networkx as nx
from collections import defaultdict



class GraphManipulation:

    def __init__(self, directed=False, G_structure='USA'):
        if directed == True:
            self.G = nx.DiGraph()
        else:
            self.G = nx.Graph()

        self.G_structure = G_structure


    def create_graph(self, df, pk_item = 'pk_article',carcinomata=None):
        '''
        Create a graph structure on the base of the given dataframe
        '''

        self.pk_item = pk_item

        # --- Extract unique users, sessions and articles from the dataframe
        users = df.pk_user.unique()
        sessions = df.pk_session.unique()
        articles = df[pk_item].unique()

        # --- Create graph nodes for three entities
        self.G.add_nodes_from(users, entity='U')
        self.G.add_nodes_from(articles, entity='A')

        for s in sessions:
            session_start = df[df['pk_session'] == s]['date-time'].min()
            self.G.add_node(s, datetime=session_start, entity='S')
        print(f'Finished Sessions')
        # --- Create edges from users to sessions with the edge type "US"
        for u in users:
            s_list = list(df[df['pk_user'] == u]['pk_session'])
            for s in s_list:
                #self.G.add_edges_from([(u, s)], edge_type='US')
                self.G.add_edge(u, s, edge_type='US')
        print(f'Finished Users')
        # --- Create edges from sessions to articles with the edge type "SA",
        # each edge has an additional attributes "reading_datetime" (when article was read)
        # and "timeview" (for how long the article was being read)
        for s in sessions:
            a_list = list(df[df['pk_session'] == s][pk_item])
            for a in a_list:
                date_time = pd.to_datetime(df[(df['pk_session'] == s) &
                                              (df[pk_item] == a)]['date-time'].values[0])
                timeview = pd.to_numeric(df[(df['pk_session'] == s) &
                                            (df[pk_item] == a)]['timeview'].values[0])
                #self.G.add_edges_from([(s, a)], edge_type='SA', reading_datetime=date_time, timeview=timeview)
                self.G.add_edge(s, a, edge_type='SA', reading_datetime=date_time, timeview=timeview)
        print(f'Finished Sessions Articles')
        if self.G_structure == 'USAC':
            categories = df.pk_category.unique()
            self.G.add_nodes_from(categories, entity='C')
            for a in articles:
                c = list(df[df[self.pk_item] == a]['pk_category'])[0]
                self.G.add_edge(a, c, edge_type='AC')
        if self.G_structure == 'CUSA':
            carcinomata_item =carcinomata.unique()
            self.G.add_nodes_from(carcinomata_item, entity='C')
            for u in users:
                c = list(df[df['pk_user'] == a]['pk_category'])[0]
                self.G.add_edge(a, c, edge_type='AC')
        print(f'USAC')

    # def add_categories_data(self, cat_df):
    #
    #     self.G_structure = 'USAC'
    #
    #     cat_nodes = list(cat_df)
    #     self.G.add_nodes_from(cat_nodes, entity='C')
    #
    #     # Create a dict from the dataframe
    #     cat_dict = cat_df.to_dict('index')
    #
    #     # For each article assign the most probable category
    #     # (later can be assigned several with different weights)
    #     article_cat = dict()
    #     for a in cat_dict:
    #         most_probable_cat = [k for k, v in cat_dict[a].items() if v == max(cat_dict[a].values())][0]
    #         article_cat[a] = most_probable_cat
    #
    #     list_of_edges = [(a, article_cat[a]) for a in article_cat]
    #
    #     # --- Create edges from articles to categories with the edge type "AC"
    #     articles = [n for n, attr in self.G.nodes(data=True) if attr['entity']=='A']
    #
    #     for (a, c) in list_of_edges:
    #         if a in articles:
    #             self.G.add_edge(a, c, edge_type='AC')
    #
    #     # --- Remove articles without categories
    #     for a in self.get_articles(self.G):
    #         if self.map_category(a) == '':
    #             self.G.remove_node(a)
    #
    #     # --- Remove sessions without articles
    #     for s in self.get_sessions(self.G):
    #         a_list = [a for a in self.G[s] if self.G[s][a]['edge_type'] == 'SA']
    #         if len(a_list) == 0:
    #             self.G.remove_node(s)
    #
    #     # --- Remove users without sessions
    #     for u in self.get_users(self.G):
    #         s_list = [s for s in self.G[u] if self.G[u][s]['edge_type'] == 'US']
    #         if len(s_list) == 0:
    #             self.G.remove_node(u)
    #
    #
    # def add_video_cat_data(self, cat_df):
    #
    #     self.G_structure = 'USAC'
    #
    #     cat_nodes = cat_df['video_category_id'].unique()
    #     self.G.add_nodes_from(cat_nodes, entity='C')
    #
    #     for a in self.get_articles(self.G):
    #         if a in cat_df.index:
    #             c = cat_df.get_value(a, 'video_category_id')
    #             self.G.add_edge(a, c, edge_type='AC')
    #         else:
    #             self.G.remove_node(a)
    #
    #     # --- Remove sessions without articles
    #     for s in self.get_sessions(self.G):
    #         a_list = [a for a in self.G[s] if self.G[s][a]['edge_type'] == 'SA']
    #         if len(a_list) == 0:
    #             self.G.remove_node(s)
    #
    #     # --- Remove users without sessions
    #     for u in self.get_users(self.G):
    #         s_list = [s for s in self.G[u] if self.G[u][s]['edge_type'] == 'US']
    #         if len(s_list) == 0:
    #             self.G.remove_node(u)


    def add_locations_data(self, loc_df):

        self.G_structure = 'USACL'

        loc_nodes = loc_df.location.unique()
        self.G.add_nodes_from(loc_nodes, entity='L')


        # For each article in the graph assign its location

        articles = self.get_articles(self.G)
        for a in articles:
            l = list(loc_df[loc_df['article'] == a]['location'])[0]
            self.G.add_edge(a, l, edge_type='AL')


    @staticmethod
    def remove_sessions_with_one_article(test_G):

        # Remove sessions that have only one article
        short_sessions = [n for n, attr in test_G.nodes(data=True)
                         if attr['entity']=='S' and test_G.degree(n) == 2] # degree 2 means only one path U-S-A exists
        test_G.remove_nodes_from(short_sessions)

        # Remove users that don't have sessions anymore
        single_users = [n for n, attr in test_G.nodes(data=True)
                        if attr['entity'] == 'U' and nx.degree(test_G, n) == 0]
        test_G.remove_nodes_from(single_users)

        # Remove articles that don't appear in any session anymore
        single_articles = [n for n, attr in test_G.nodes(data=True)
                           if attr['entity'] == 'A'
                           and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
        test_G.remove_nodes_from(single_articles)

        return test_G

    @staticmethod
    def filter_meaningless_sessions(g, timeview):

        sessions = [s for s,attr in g.nodes(data=True) if attr['entity']=='S']

        for session in sessions:
            timeviews = [attr['timeview'] for s, a, attr in g.edges(data=True)
                         if attr['edge_type'] == 'SA' and (s==session or a==session)]
            if all([t<=timeview for t in timeviews]):
                # print(session, timeviews)
                g.remove_node(session)

        # Remove users that don't have sessions anymore
        single_users = [n for n, attr in g.nodes(data=True)
                        if attr['entity'] == 'U' and nx.degree(g, n) == 0]
        g.remove_nodes_from(single_users)

        # Remove articles that don't appear in any session anymore
        single_articles = [n for n, attr in g.nodes(data=True)
                           if attr['entity'] == 'A'
                           and all([g[n][m]['edge_type'] != 'SA' for m in g[n]])]
        g.remove_nodes_from(single_articles)

        return g





    @staticmethod
    def filter_sessions(test_G, n_items = 1):

        # Remove sessions that have only one article
        short_sessions = [n for n, attr in test_G.nodes(data=True)
                          if attr['entity'] == 'S' and test_G.degree(n) <= (n_items + 1)] # (one edge goes to user)
        test_G.remove_nodes_from(short_sessions)

        # Remove users that don't have sessions anymore
        # After filtering out the sessions with less than min items, we remove also the users
        # who have left without any corresponding session
        single_users = [n for n, attr in test_G.nodes(data=True)
                        if attr['entity'] == 'U' and nx.degree(test_G, n) == 0]
        test_G.remove_nodes_from(single_users)

        # Remove articles that don't appear in any session anymore
        # Same as the users for articles
        single_articles = [n for n, attr in test_G.nodes(data=True)
                           if attr['entity'] == 'A'
                           and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
        test_G.remove_nodes_from(single_articles)

        return test_G

    @staticmethod
    def filter_timeviews(test_G, timeview=2):

        # Remove sessions that have only one article
        short_timeviews = [(s, a) for s, a, attr in test_G.edges(data=True)
                           if attr['edge_type'] == 'SA' and attr['timeview'] < timeview]
        test_G.remove_edges_from(short_timeviews)

        # Remove articles that don't appear in any session anymore
        single_articles = [n for n, attr in test_G.nodes(data=True)
                           if attr['entity'] == 'A'
                           and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
        test_G.remove_nodes_from(single_articles)

        # Remove users that don't have sessions anymore
        single_users = [n for n, attr in test_G.nodes(data=True)
                        if attr['entity'] == 'U' and nx.degree(test_G, n) == 0]
        test_G.remove_nodes_from(single_users)

        return test_G


    @staticmethod
    def filter_users(test_G, n_sessions = 1):

        # Remove users that have less than specified number of sessions
        inactive_users = [n for n, attr in test_G.nodes(data=True)
                          if attr['entity'] == 'U' and test_G.degree(n) < n_sessions]
        test_G.remove_nodes_from(inactive_users)

        # Remove sessions that don't have user anymore
        single_sessions = [n for n, attr in test_G.nodes(data=True)
                           if attr['entity'] == 'S' and nx.degree(test_G, n) == 0]
        test_G.remove_nodes_from(single_sessions)

        # Remove articles that don't appear in any session anymore
        single_articles = [n for n, attr in test_G.nodes(data=True)
                           if attr['entity'] == 'A'
                           and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
        test_G.remove_nodes_from(single_articles)

        return test_G


    @staticmethod
    def remove_users_that_werent_active_in_train(test_G, train_G):

        train_users = [n for n, attr in train_G.nodes(data=True) if attr['entity']=='U']
        test_users = [n for n, attr in test_G.nodes(data=True) if attr['entity']=='U']

        only_test_active_users = [n for n in test_users if n not in train_users]

        sessions_to_remove = []
        for u in only_test_active_users:
            u_sessions = [n for n, attr in test_G.nodes(data=True)
                          if attr['entity']=='S'
                          and nx.has_path(test_G, u, n) == True
                          and nx.shortest_path_length(test_G, u, n) == 1]

            sessions_to_remove.extend(u_sessions)

        test_G.remove_nodes_from(only_test_active_users)
        test_G.remove_nodes_from(sessions_to_remove)

        single_articles = [a for a,attr in test_G.nodes(data=True)
                           if attr['entity']=='A' and nx.degree(test_G, a) == 0] # Will not work with having categories !

        test_G.remove_nodes_from(single_articles)

        return test_G


    @staticmethod
    def remove_items_that_didnt_exist_in_train(test_G, train_G):

        train_items = [n for n, attr in train_G.nodes(data=True) if attr['entity'] == 'A']
        test_items = [n for n, attr in test_G.nodes(data=True) if attr['entity'] == 'A']

        only_test_items = [n for n in test_items if n not in train_items]

        test_G.remove_nodes_from(only_test_items)

        return test_G

    # def extract_subgaph_given_time_interval(self, start_time, end_time):
    #
    #     sessions = [n for n,attr in self.G.nodes(data=True)
    #                 if attr['entity'] == 'S'
    #                 and attr['datetime'] >= start_time
    #                 and attr['datetime'] < end_time]
    #
    #     sub_G = nx.Graph()
    #     for s in sessions:
    #         nodes = nx.neighbors(self.G, s)
    #         nodes.append(s)
    #         temp_sub_G = self.G.subgraph(nodes)
    #         sub_G = nx.compose(sub_G, temp_sub_G)
    #
    #     return sub_G

    @staticmethod
    def derive_adjacency_multigraph(G, entity1, entity2, path_len=2):
        """
        Derive a "bipartite" graph from a heterogenuos graph

        Parameters
        -----------
        entity1 : character
        entity2 : character
        path_len : the length of the path needed to go from entity1 to entity2

        Returns
        -----------
        G_new : a new multi-graph, consising of only two specified entities and multiple links between them
        """

        # --- Initialization
        adj = defaultdict(list)
        nodes1 = [(n,attr) for n,attr in G.nodes(data=True) if attr['entity'] == entity1]
        nodes2 = [(n,attr) for n,attr in G.nodes(data=True) if attr['entity'] == entity2]

        # --- Building adjacency matrix
        for n1, _ in nodes1:
            adj[n1] = defaultdict(list)
            for n2, _ in nodes2:
                adj[n1][n2] = len([path for path in nx.all_simple_paths(G, n1, n2, path_len)])

        # --- Building a multi-graph on the base of adjacency matrix

        G_new = nx.MultiGraph()

        G_new.add_nodes_from(nodes1)#, entity=entity1)
        G_new.add_nodes_from(nodes2)#, entity=entity2)

        for n1 in adj:
            for n2 in adj[n1]:
                for i in range(adj[n1][n2]):
                    G_new.add_edges_from([(n1, n2)])

        return G_new


    @staticmethod
    def get_users(g,entity = 'U'):
        return [n for n, attr in g.nodes(data=True) if attr['entity'] == entity]

    @staticmethod
    def get_sessions(g):
        return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'S']

    @staticmethod
    def get_articles(g):
        return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'A']

    @staticmethod
    def get_categories(g):
        return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'C']

    @staticmethod
    def get_locations(g):
        return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'L']

    @staticmethod
    def get_nodes(g, entity):
        return [n for n, attr in g.nodes(data=True) if attr['entity'] == entity]

    @staticmethod
    def get_sessions_per_user(g,entity = 'U'):

        users = [n for n, attr in g.nodes(data=True) if attr['entity'] == entity]
        sessions_per_user = []
        for u in users:
            sessions_per_user.append(len(g[u]))

        return sessions_per_user

    @staticmethod
    def get_articles_per_session(g,entity = 'S'):

        sessions = [n for n, attr in g.nodes(data=True) if attr['entity'] == entity]

        articles_per_session = []
        for s in sessions:
            articles_per_session.append(nx.degree(g, s)-1) # -1 because one edge goes to the user

        return articles_per_session

    @staticmethod
    def create_sac_graph(g):

        user_nodes = [n for n, attr in g.nodes(data=True) if attr['entity'] == 'U']
        g.remove_nodes_from(user_nodes)

        return g

    @staticmethod
    def create_sa_graph(g):

        new_g = g.copy()
        user_category_nodes = [n for n, attr in g.nodes(data=True) if attr['entity'] in ['U','C']]
        new_g.remove_nodes_from(user_category_nodes)

        return new_g

    @staticmethod
    def create_subgraph_of_adjacent_entities(g, list_of_entities):

        new_g = g.copy()
        nodes = [n for n, attr in g.nodes(data=True) if attr['entity'] not in list_of_entities]
        new_g.remove_nodes_from(nodes)

        return new_g

    @staticmethod
    def derive_adjacency_multigraph(G, entity1, entity2, path_len=2):
        """
        Derive a "bipartite" graph from a heterogenous graph

        Parameters
        -----------
        entity1 : character - target entity
        entity2 : character - destination entity
        path_len : the length of the path needed to go from entity1 to entity2

        Returns
        -----------
        G_new : a new multi-graph, consising of only two specified entities and multiple links between them

        """

        # --- Initialization
        adj = defaultdict(list)
        nodes1 = [n for n, attr in G.nodes(data=True) if attr['entity'] == entity1]
        nodes2 = [n for n, attr in G.nodes(data=True) if attr['entity'] == entity2]

        # --- Building adjacency matrix
        for n1 in nodes1:
            adj[n1] = defaultdict(list)
            for n2 in nodes2:
                adj[n1][n2] = len([path for path in nx.all_simple_paths(G, n1, n2, path_len)])

        # --- Building a multi-graph on the base of adjacency matrix

        G_new = nx.Graph()

        G_new.add_nodes_from(nodes1, entity=entity1)
        G_new.add_nodes_from(nodes2, entity=entity2)

        for n1 in adj:
            for n2 in adj[n1]:
                for i in range(adj[n1][n2]):
                    G_new.add_edges_from([(n1, n2)],edge_type = f'{entity1}{entity2}')

        return G_new

    def map_category(self, a):

        c = [c for c in self.G[a] if self.G[a][c]['edge_type'] == 'AC']
        c = c[0] if len(c) > 0 else ''

        return c

    def map_location(self, a):

        l = [l for l in self.G[a] if self.G[a][l]['edge_type'] == 'AL']
        l = l[0] if len(l) > 0 else ''

        return l

    @staticmethod
    def map_timeview(g, session, article):

        t = [attr['timeview'] for s, a, attr in g.edges(data=True) if (s==session and a==article) or (s==article and a==session)][0]

        return t

    def map_enity(self, n):

        entity = self.G.node[n]['entity']

        return entity


    def count_unique_categories(self, rec):

        c_list = []
        for a in rec:
            c = [c for c in self.G[a] if self.G[a][c]['edge_type'] == 'AC']
            if len(c) > 0:
                c_list.append(c[0])
            #print(a, c)
            #continue
            #c_list.append(c)

        return len(list(set(c_list)))

    @staticmethod
    def get_active_users(g, n_sessions = 1):

        active_users = [n for n, attr in g.nodes(data=True)
                             if attr['entity'] == 'U' and g.degree(n) >= n_sessions]

        return active_users
