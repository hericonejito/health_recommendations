import networkx as nx
from operator import itemgetter

class PopularityBasedRec:

    def __init__(self, G=None, N=1):
        self.G = G
        self.N = N

    def compute_pop(self, train_G):
        '''
        Given the train data of users articles interaction
        return the ordered list of the most popular articles to recommend
        #(both considering timeview or only interactions) - not implemented, now only considering interactions
        '''

        self.train_G = train_G # data to compute popularity on

        # !!! DEGREE ONLY WORKS WHEN ARTICLES RE ONLY CONNECTED WITH SESSIONS. WHEN ADDING CATEGORIES WON'T WORK ANYMORE
        view_items_freq = [(n, self.train_G.degree(n)-2)
                           for n, attr in self.train_G.nodes(data=True)
                           if attr['entity']=='A']

        # comment test Lidija
        # articles = [n for n,attr in train_G.nodes(data=True) if attr['entity']=='A']
        # view_items_freq = []
        # for a in articles:
        #     degree = len([s for s in train_G[a] if train_G[a][s]['edge_type'] != 'SA'])
        #     view_items_freq.append((a, degree))

        # sorted_view_items_freq = sorted(view_items_freq, key=lambda x: x[1], reverse=True)

        view_items_freq_dict = dict(view_items_freq)
        sorted_view_items_freq = [(k, v) for k, v in sorted(view_items_freq_dict.items(), key=itemgetter(1), reverse=True)]

        # max([i[1] for i in sorted_view_items_freq])

        # print('Items view frequencies:', sorted_view_items_freq, '\n')
        self.pop_items_list = [i[0] for i in sorted_view_items_freq]

        self.pop_items_and_cat_list = []
        for i, d in sorted_view_items_freq:
            for n in self.train_G[i]:
                if (self.train_G[i][n]['edge_type'] == 'AC'):
                    self.pop_items_and_cat_list.append((i, n, d))


    def construct_rec_list(self):
        '''
        Function to construct the rec list for each active user in the train set in the form (user_id, user_rec)
        based on the most popular items not yet viewed by the user
        '''
        # QUESTION: With popularity algorithm we can recommend not only for users that were active in train period,
        # but for all of them, as long as we just recommend the most popular item of the period to all users!
        # Right?

        users_list = [n for n, attr in self.train_G.nodes(data=True)
                      if attr['entity']=='U'] # set of users for which provide a recommendation
        self.rec_list = list()

        for user in users_list:
            user_rec = []
            i = 0
            for pop_item in self.pop_items_list:
                if i < self.N:
                    # if user has already read this article
                    if len(nx.shortest_path(self.train_G, source=user, target=pop_item)) == 3:
                        continue
                    else:
                        user_rec.append(pop_item)
                        i += 1
            self.rec_list.append((user, user_rec))


    def predict_next(self, user, item_list, remind=False, cat_list=[]):
        '''
        Given user and the list of already viewed items in the test session predict a next item
        returning a ranked list of N predictions
        '''

        pop_items_list = []
        if cat_list == []:
            pop_items_list = self.pop_items_list
        else:
            for cat in cat_list:
                cat_item_pop_list = [(i, d) for i, c, d in self.pop_items_and_cat_list if c == cat]
                sorted_cat_item_pop_list = sorted(cat_item_pop_list, key=lambda x: x[1], reverse=True)
                pop_items_list.extend([i[0] for i in sorted_cat_item_pop_list])


        user_rec = []
        i = 0
        for pop_item in pop_items_list:
            if i < self.N:
                # if user was active in train period and has already read this article
                # MAYBE WE CAN ALSO PUT REMINDERS (NOT EXCLUDE IF A USER ALREADY READ IT IN TRAIN)
                # if ((user in self.train_G) \
                #        and (nx.has_path(self.train_G, user, pop_item) == True) \
                #        and (len(nx.shortest_path(self.train_G, source=user, target=pop_item)) == 3)) \
                #        or (pop_item in item_list):
                #    continue
                # else:
                #     user_rec.append(pop_item)
                #     i += 1

                if pop_item in item_list:
                    continue
                else:
                    if remind == False:
                        # We don't recommend articles that user has read during train period
                        if ((user in self.train_G)
                            and (nx.has_path(self.train_G, user, pop_item) == True)
                            and (len(nx.shortest_path(self.train_G, source=user, target=pop_item)) == 3)):
                           continue
                        else:
                            user_rec.append(pop_item)
                            i += 1
                    else:
                        user_rec.append(pop_item)
                        i += 1

                # We don't care if user was in a training set or not, and if he already watched the article or not,
                # we just recommend the most popular
                #user_rec.append(pop_item)
                #i += 1

        return user_rec
