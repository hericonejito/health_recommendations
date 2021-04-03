from collections import defaultdict
import numpy as np
import math

class AccuracyEvaluation:

    def __init__(self, G,number_recommendations = 1,edge_type='edge_type'):
        self.n_recommendations = number_recommendations
        self.G = G
        self.edge_type = edge_type
        self.methods_list = set()

        self.rec_precision = defaultdict(list) # if the article is in the rec list, then it is a hit (1)
        self.rec_ndcg = defaultdict(list) # captures the position of the article in the rec list: 1/log_2(pos)
        self.rec_diversity = defaultdict(list) # ILD index (intra-list diversity): average pairwise "distance" of the items in the rec list
        self.rec_explainability = defaultdict(list)
        self.rec_recall = defaultdict(list)

        self.session_precision = defaultdict(list)
        self.session_ndcg = defaultdict(list)
        self.session_diversity = defaultdict(list)
        self.session_explainability = defaultdict(list)
        self.session_recall = defaultdict(list)

        self.tw_precision = defaultdict(list)
        self.tw_ndcg = defaultdict(list)
        self.tw_diversity = defaultdict(list)
        self.tw_explainability = defaultdict(list)
        self.tw_recall = defaultdict(list)

        self.precision = defaultdict(float) # = hit
        self.ndcg = defaultdict(float)
        self.diversity = defaultdict(float)
        self.explainability = defaultdict(float)
        self.recall = defaultdict(float)

        self.explainability_matrix = None


    def get_rec_diversity(self, rec):

        if len(rec) == 1:
            return 0

        c_list = []
        for a in rec:
            c = [c for c in self.G[a] if self.G[a][c][self.edge_type] == 'AC']
            if len(c) > 0:
                c_list.append(c[0])

        pairwise_distance = 0
        for i, c1 in enumerate(c_list):
            for j, c2 in enumerate(c_list):
                if i != j:
                    if c1 != c2:
                        pairwise_distance += 1

        ild = pairwise_distance / (len(rec) * (len(rec) - 1)) if len(rec) > 1 else 0

        return ild


    def evaluate_recommendation(self, rec, truth, method, s=None):
        '''
        For the built recommendation measure its precision, ndcg and diversity
        '''

        # if len(rec) == 0:
        #     return

        if method not in self.methods_list:
            self.methods_list.add(method)

            self.rec_precision[method] = []
            self.rec_ndcg[method] = []
            self.rec_diversity[method] = []
            self.rec_explainability[method] = []
            self.rec_recall[method] = []
            self.session_precision[method] = []
            self.session_ndcg[method] = []
            self.session_diversity[method] = []
            self.session_explainability[method] = []
            self.session_recall[method] = []
            self.tw_precision[method] = []
            self.tw_ndcg[method] = []
            self.tw_diversity[method] = []
            self.tw_explainability[method] = []
            self.tw_recall[method] = []


        precision = 1/self.n_recommendations if truth in rec else 0
        ndcg = 1/math.log(1+(rec[truth]+1), 2) if truth in rec else 0
        #hit_score = 1 / (rec.index(truth) + 1) if truth in rec else 0 # another possible measure similar to ndcg, not used
        diversity = self.get_rec_diversity(rec)

        if s != None and self.explainability_matrix != None and len(self.explainability_matrix)>0:
            expl_scores = [self.explainability_matrix[s][a] if a in self.explainability_matrix[s] else 0 for a in rec]
            if len(expl_scores) > 0:
                explainability = np.mean(expl_scores)
                self.rec_explainability[method].append(explainability)
            # print(method, self.rec_explainability[method], self.explainability_matrix[s])

        self.rec_precision[method].append(precision)
        self.rec_ndcg[method].append(ndcg)
        self.rec_diversity[method].append(diversity)
        self.rec_recall[method].append(1)


    def evaluate_session(self):

        for method in self.methods_list:

            if len(self.rec_precision[method]) == 0:
                return

            self.session_precision[method].append(np.mean([v for v in self.rec_precision[method]]))
            self.session_ndcg[method].append(np.mean([v for v in self.rec_ndcg[method]]))
            self.session_diversity[method].append(np.mean([v for v in self.rec_diversity[method]]))
            total_truths = np.sum([v for v in self.rec_recall[method]])
            total_correct = np.sum([v for v in self.rec_precision[method]])*self.n_recommendations
            self.session_recall[method].append(total_correct / total_truths)
            if self.explainability_matrix != None and len(self.rec_explainability[method]) != 0:
                # print(method, self.rec_explainability[method])
                self.session_explainability[method].append(np.mean([v for v in self.rec_explainability[method]]))

            # clean recs evaluation list for the next session
            self.rec_precision[method] = []
            self.rec_ndcg[method] = []
            self.rec_diversity[method] = []
            self.rec_explainability[method] = []
            self.rec_recall[method] = []

    def evaluate_tw(self):

        for method in self.methods_list:

            if len(self.session_precision[method]) == 0:
                return

            self.tw_precision[method].append(np.mean([v for v in self.session_precision[method]]))
            self.tw_ndcg[method].append(np.mean([v for v in self.session_ndcg[method]]))
            self.tw_diversity[method].append(np.mean([v for v in self.session_diversity[method]]))
            self.tw_recall[method].append(np.mean([v for v in self.session_recall[method]]))
            if self.explainability_matrix != None and len(self.session_explainability[method]) != 0:
                self.tw_explainability[method].append(np.mean([v for v in self.session_explainability[method]]))

            # clean sessions evaluation list for the next time window
            self.session_precision[method] = []
            self.session_ndcg[method] = []
            self.session_diversity[method] = []
            self.session_explainability[method] = []
            self.session_recall[method] = []


    def evaluate_total_performance(self):

        for method in self.methods_list:

            if len(self.tw_precision[method]) == 0:
                return

            self.precision[method] = np.mean([v for v in self.tw_precision[method]])
            self.ndcg[method] = np.mean([v for v in self.tw_ndcg[method]])
            self.diversity[method] = np.mean([v for v in self.tw_diversity[method]])
            self.recall[method] = np.mean([v for v in self.tw_recall[method]])
            if self.explainability_matrix != 0:
                self.explainability[method] = np.mean([v for v in self.tw_explainability[method]])



