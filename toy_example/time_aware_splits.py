import pandas as pd
import datetime
import networkx as nx


class TimeAwareSplits:

    def __init__(self, G,session_entity_prefix = 'S'):
        self.G = G
        self.session_entity_prefix = session_entity_prefix
        self.time_splits_df_list = []
        self.time_splits_graph_list = []
        self.time_window_graph_list = []

    def create_time_split_graphs(self, G, num_splits=12):
        '''
        Given the USA Graph and a number of splits
        return a list of graphs that are a sub-samples based on time of the original one
        '''
        self.G = G

        session_times = [attr['datetime'] for n, attr in self.G.nodes(data=True) if attr['entity']==self.session_entity_prefix]

        starting_time = min(session_times)
        ending_time = max(session_times)
        time_delta = (ending_time - starting_time) / num_splits

        time_span_list = []
        for i in range(num_splits):
            t_i = starting_time + i * time_delta
            t_f = t_i + time_delta
            time_span_list.append((t_i, t_f))

        self.time_span_list = time_span_list

        # --- Works when graph is of struture USA and has no categories
        # for time_span in time_span_list:
        #     temp_sessions = [n for n, attr in self.G.nodes(data=True)
        #                      if attr['entity']=='S'
        #                      and attr['datetime'] >= time_span[0]
        #                      and attr['datetime'] < time_span[1] + datetime.timedelta(0,0.1)]
        #
        #     temp_G = nx.Graph()
        #     for s in temp_sessions:
        #         nodes = nx.neighbors(self.G, s)
        #         nodes.append(s)
        #         sub_G = self.G.subgraph(nodes)
        #         temp_G = nx.compose(temp_G, sub_G)
        #
        #     self.time_splits_graph_list.append(temp_G)

        # For a graph structure USAC : (TEST)
        for time_span in time_span_list:
            temp_sessions = [n for n, attr in self.G.nodes(data=True)
                             if attr['entity'] == self.session_entity_prefix
                             and attr['datetime'] >= time_span[0]
                             and attr['datetime'] < time_span[1] + datetime.timedelta(0, 0.1)]

            temp_neighbors = []
            for s in temp_sessions:
                temp_neighbors.extend(nx.neighbors(self.G, s))

            temp_neighbors = list(set(temp_neighbors))

            categories = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == 'C']

            temp_nodes = []
            temp_nodes.extend(temp_sessions)
            temp_nodes.extend(temp_neighbors)
            temp_nodes.extend(categories)

            temp_G = self.G.subgraph(temp_nodes)

            self.time_splits_graph_list.append(temp_G)


    def create_time_window_graphs(self, window_size=1):
        '''
        Given the list of graphs splitted by time
        return a list of tuple (long_train_set, short_train_set, test_set) with train dataset as the concatenation
        of window_size time split graphs and the test set as the next time split graph
        '''
        num_splits = len(self.time_splits_graph_list)
        for i in range(window_size, num_splits):

            long_train_G = nx.Graph()
            long_train_set_list = self.time_splits_graph_list[:i]
            for g in long_train_set_list:
                long_train_G = nx.compose(long_train_G, g)

            # short_train_G = nx.Graph()
            # short_train_set_list = self.time_splits_graph_list[i - window_size:i]
            # for g in short_train_set_list:
            #     short_train_G = nx.compose(short_train_G, g)

            test_G = self.time_splits_graph_list[i]

            # self.time_window_graph_list.append((long_train_G, short_train_G, test_G))
            start_time = self.time_span_list[i][0]
            end_time = self.time_span_list[i][1]
            self.time_window_graph_list.append((long_train_G, test_G,start_time,end_time))

    def create_short_term_train_set(self, test_session_start, back_timedelta, test_session_graph=None):

        train_start = test_session_start - back_timedelta
        train_end = test_session_start

        temp_sessions = [n for n, attr in self.G.nodes(data=True)
                         if attr['entity'] == 'S'
                         and attr['datetime'] >= train_start
                         and attr['datetime'] < train_end]

        # temp_neighbors = []
        # for s in temp_sessions:
        #     temp_neighbors.extend(nx.neighbors(self.G, s))
        #
        # temp_neighbors = list(set(temp_neighbors))
        #
        # categories = [n for n, attr in self.G.nodes(data=True) if attr['entity'] == 'C']

        temp_users = []
        temp_articles = []
        for s in temp_sessions:
            temp_users.extend([u for u in self.G[s] if self.G[s][u]['edge_type'] == 'US'])
            temp_articles.extend([a for a in self.G[s] if self.G[s][a]['edge_type'] == 'SA'])

        temp_users = list(set(temp_users))
        temp_articles = list(set(temp_articles))

        temp_categories = []
        for a in temp_articles:
            temp_categories.extend([c for c in self.G[a] if self.G[a][c]['edge_type'] == 'AC'])

        temp_categories = list(set(temp_categories))

        temp_locations = []
        for a in temp_articles:
            temp_locations.extend([l for l in self.G[a] if self.G[a][l]['edge_type'] == 'AL'])

            temp_locations = list(set(temp_locations))

        temp_nodes = []
        temp_nodes.extend(temp_sessions)
        # temp_nodes.extend(temp_neighbors)
        # temp_nodes.extend(categories)
        temp_nodes.extend(temp_users)
        temp_nodes.extend(temp_articles)
        temp_nodes.extend(temp_categories)
        temp_nodes.extend(temp_locations)

        short_train_subgraph = self.G.subgraph(temp_nodes)

        if test_session_graph != None:
            short_train_subgraph = nx.compose(short_train_subgraph, test_session_graph)

        return short_train_subgraph


    def create_long_term_user_train_set(self, user, session, s_datetime, articles, recent_users):

        long_user_g = self.G.copy()

        # Remove sessions that appeared later than the current user session
        future_sessions = [n for n, attr in long_user_g.nodes(data=True)
                           if attr['entity'] == 'S'
                           and attr['datetime'] > s_datetime]
        long_user_g.remove_nodes_from(future_sessions)

        # Remove articles from the current user session that he has not read yet
        future_articles = [(session, a) for a in long_user_g[session]
                           if long_user_g[session][a]['edge_type'] == 'SA' and a not in articles]
        long_user_g.remove_edges_from(future_articles)

        # Remove users that do not appear in recent short term
        long_user_g.remove_nodes_from([n for n, attr in long_user_g.nodes(data=True)
                                       if (attr['entity']=='U') and (n not in recent_users) and (n != user)])

        # Remove sessions that don't have user anymore
        single_sessions = [n for n, attr in long_user_g.nodes(data=True)
                           if attr['entity'] == 'S' and nx.degree(long_user_g, n) == 0]
        long_user_g.remove_nodes_from(single_sessions)

        # Remove articles that don't appear in any session anymore
        single_articles = [n for n, attr in long_user_g.nodes(data=True)
                           if attr['entity'] == 'A'
                           and all([long_user_g[n][m]['edge_type'] != 'SA' for m in long_user_g[n]])]
        long_user_g.remove_nodes_from(single_articles)

        # --- Extract a subgraph with only users connected to current user
        long_user_g = long_user_g.subgraph(nx.node_connected_component(long_user_g, user))

        return long_user_g




