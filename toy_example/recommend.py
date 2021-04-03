#!/usr/bin/python3.6
import os
#import gunicorn
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State,Event

import time

from rq import Queue
from worker import conn
import uuid

from data_import import *
from graph_manipulation import *
from time_aware_splits import *
from popularity_based_rec import *
from personalized_prank import *
from pathsim import *
from simrank import *
from accuracy_evaluation import *
from collections import defaultdict
from collections import Counter
from operator import itemgetter
from background_jobs import create_graph
import pandas as pd
import networkx as nx
from collections import defaultdict
import plotly.graph_objs as go
from datetime import datetime
import numpy as np

import datetime

e = ""




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
datasets = [('test','test'),('German News Provider','german_news'),('Italian News Provider','italian_news'),('German TVBroadcasts Provider','german_tvbroadcasts')]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([
    html.Div(id='job_id',style={'display':'none'}),

    # dcc.Input(id='my-id', value='initial value', type='text'),
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # in milliseconds
        n_intervals=100
    ),
html.Div([
        html.Center('Comparison of Graph-based algorithms for Session-based Recommendations', style={'color':'blue','font-weight':'bold'})]),
    
    html.Br(),
    html.Div([
        html.Center('Co-creators: P. Symeonidis, L. Kirjackaja, S. Chairistanidis, and M. Zanker', style={'color':'blue','font-weight':'bold','font-size':12})]),
    html.Br(),
    html.Div([
        html.Div('Select a dataset'),
        dcc.Dropdown(
            id='xaxis-column',
            options=[{'label': i[0], 'value': i[1]} for i in datasets],
            value='italian_news'

        )]),
    html.Div([

        html.Div('Number of Time Splits'),

        dcc.Input(id='number_of_splits',
                  placeholder='Number Of Splits',
                  type='number',
                  min=1,
                  step=1,
                  value=12,

                  ),


        html.Div('Time Window Size'),

        dcc.Input(id='short_days',
                  placeholder='Short Days Window',
                  type='number',
                  min=1,
                  step=1,
                  value=1,

                  ),

        html.Div('Number of Recommendations'),

        dcc.Input(id='number_recommendations',
                  placeholder='Number Of Recommendations',
                  type='number',
                  min=1,
                  step=1,
                  value=1,
                  ),

        html.Div('Minimum Number of Items per Session'),

        dcc.Input(id='min_items_n',
                  placeholder='Minimum Number of Items per Session',
                  type='number',
                  min=1,
                  step=1,
                  value=3,

                  )

    ],style={'columnCount': 2}),

    html.Br(),
    html.Label('Node types',style={'font-weight':'bold'}),
    html.Div([
    dcc.Checklist(id='nodes',
        options=[
            {'label': 'Users', 'value': 'U'},
            {'label': 'Articles', 'value': 'A'},
            {'label': 'Sessions', 'value': 'S'},
            {'label': 'Locations', 'value': 'L'},
            {'label': 'Categories', 'value': 'C'},
        ],
        values=['U','S', 'A']
    ),],style={'columnCount': 3}),
    html.Br(),
    html.Label('Methods Comparison',style={'font-weight':'bold'}),

    html.Div([
    dcc.Checklist(id='methods',
        options=[
            {'label': 'Pop', 'value': 'POP'},
            {'label': 'RWR', 'value': 'RWR'},
            {'label': 'Simrank', 'value': 'Simrank'},
            {'label': 'Pathsim', 'value': 'Pathsim'},
            {'label':'PathCount','value': 'PathCount'}
            # {'label': 'SKNN', 'value': 'SKNN'}
        ],
        values=['RWR','POP']
    ),],style={'columnCount': 3}),
    html.Br(),
    html.Button('Execute',id='execute'),
    html.Button('Toggle Details',id='hide'),
    html.Br(),
    html.Div(id='status_wrap',children=[dcc.Markdown(id='status')]),

    html.Div([

        html.Div(children='', id='dummy-results'),
        dcc.Interval(
            id='update-interval',
            interval=60 * 60 * 5000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Graph(id = 'evaluation'),



    ], id='results'),
],)


# @app.callback(
#     Output(component_id='my-div', component_property='children'),
#     [Input(component_id='my-id', component_property='value')]
# )
# def update_output_div(input_value):
#     return 'You\'ve entered "{}"'.format(input_value)
# @app.callback(
#     dash.dependencies.Output('job_id', 'children'),
#     [dash.dependencies.Input('execute', 'n_clicks')],
#
#  state = [State('xaxis-column', 'value'),
#            State('number_of_splits', 'value'),
#            State('short_days', 'value'),
#            State('number_recommendations', 'value'),
#            State('min_items_n', 'value'),
#            State('nodes', 'values'),
#            State('methods', 'values'),
#            ]
# )
# def query_submitted(click,data_path,number_splits,short_days,number_recommendations,min_items_n,nodes,methods):
#     if click == 0 or click is None:
#         return ''
#     else:
#         # a query was submitted, so queue it up and return job_id
#         duration = 20           # pretend the process takes 20 seconds to complete
#         q = Queue(connection=conn)
#         job_id = str(uuid.uuid4())
#         print(f'Job ID when started {job_id}')
#         job = q.enqueue_call(func=create_graph,
#                                 args=(data_path,number_splits,short_days,number_recommendations,min_items_n,nodes,methods),
#
#                                 job_id=job_id)
#         return job_id
#
#
#
# @app.callback(
#     dash.dependencies.Output('update-interval', 'interval'),
#     [dash.dependencies.Input('job_id', 'children'),
#     dash.dependencies.Input('update-interval', 'n_intervals')])
# def stop_or_start_table_update(job_id, n_intervals):
#     q = Queue(connection=conn)
#
#     job = q.fetch_job(job_id)
#     if job is not None:
#         # the job exists - try to get results
#         result = job.result
#         if result is None:
#             # a job is in progress but we're waiting for results
#             # therefore regular refreshing is required.  You will
#             # need to fine tune this interval depending on your
#             # environment.
#             return 1000
#         else:
#             # the results are ready, therefore stop regular refreshing
#             return 60*60*1000
#     else:
#         # the job does not exist, therefore stop regular refreshing
#         return 60*60*1000
#
#
# # this callback checks if the job result is ready.  If it's ready
# # the results return to the table.  If it's not ready, it pauses
# # for a short moment, then empty results are returned.  If there is
# # no job, then empty results are returned.
# @app.callback(
#     dash.dependencies.Output('evaluation', 'figure'),
#     [dash.dependencies.Input('update-interval', 'n_intervals')],
#     [dash.dependencies.State('job_id', 'children')])
# def update_results_tables(n_intervals, job_id):
#     q = Queue(connection=conn)
#     print(job_id)
#     print(q.count)
#     job = q.fetch_job(job_id)
#     if job is not None:
#         # job exists - try to get result
#         result = job.result
#         print(f'Result : {result}')
#         if result is None:
#             # results aren't ready, pause then return empty results
#             # You will need to fine tune this interval depending on
#             # your environment
#             time.sleep(3)
#             return {
#         'data': go.Scatter(x=[], y=[]),
#         'layout': go.Layout(
#             xaxis={
#                 'title': 'Time_Period',
#
#             },
#             yaxis={
#                 'title': 'Precision',
#
#             })
#
#     }
#         if result is not None:
#             # results are ready
#             traces = []
#             for method in result.keys():
#                 trace = go.Scatter(x=result[method][0], y=result[method][1], name=method)
#                 traces.append(trace)
#             return        {
#                     'data': traces,
#                     'layout': go.Layout(
#                         xaxis={
#                             'title': 'Time_Period',
#
#                         },
#                         yaxis={
#                             'title': 'Precision',
#
#                         })
#
#                 }
#     else:
#         # no job exists with this id
#         return {
#         'data': go.Scatter(x=[], y=[]),
#         'layout': go.Layout(
#             xaxis={
#                 'title': 'Time_Period',
#
#             },
#             yaxis={
#                 'title': 'Precision',
#
#             })
#
#     }
#
# @app.callback(
#     dash.dependencies.Output('status', 'children'),
#     [dash.dependencies.Input('job_id', 'children'),
#     dash.dependencies.Input('update-interval', 'n_intervals')])
# def stop_or_start_table_update(job_id, n_intervals):
#     q = Queue(connection=conn)
#     job = q.fetch_job(job_id)
#     if job is not None:
#         # the job exists - try to get results
#         result = job.result
#         if result is None:
#             # a job is in progress and we're waiting for results
#             global e
#             return 'Running query.  This might take a moment - don\'t close your browser! ' + str(e)
#         else:
#             # the results are ready, therefore no message
#             return ''
#     else:
#         # the job does not exist, therefore no message
#         return ''
#


@app.callback(Output(component_id='status',component_property='children'),
              events=[Event('interval-component','interval')])
def update_text_area():
    global e
    return e

@app.callback(Output(component_id='status_wrap',component_property='style'),
              [Input('hide','n_clicks')])
def toggle_details(n_clicks):
    if n_clicks==None:
        return {'white-space':'pre-wrap','margin-top':'40px'}
    else:
        if n_clicks%2==0:
            return {'display':'block','white-space':'pre-wrap','margin-top':'40px'}
        else:
            return {'display':'none'}

@app.callback(
    Output(component_id='evaluation', component_property='figure'),
    [Input('execute', 'n_clicks')
     ],state=[  State('xaxis-column','value'),
                State('number_of_splits','value'),
                State('short_days','value'),
                State('number_recommendations','value'),
                State('min_items_n','value'),
                State('nodes','values'),
                State('methods','values'),
              ]
)
def update_div(n_clicks,data_path,number_splits,short_days,number_recommendations,min_items_n,nodes,methods):
    import itertools
    if n_clicks == None:
        global e
        e+=(f'{n_clicks}')
    else:

        # DATA_PATH = f'./Data/{data_path} - pk_client, pk_session, pk_article, timeview (s), date, time.txt'
        # CAT_DATA_PATH = f'./Data/{data_path}-5topics-doc-topics.txt'
        # LOC_DATA_PATH = f'./Data/{data_path} - pk_article, pk_district.txt'
        # gm = GraphManipulation()
        # di = DataImport()
        # di.import_user_click_data(DATA_PATH, adjust_pk_names=True)
        # print('Sterguis')
        # # --- Reduce dataset to 1 month / 1 week / ...
        # # di.reduce_timeframe(dt.datetime(2017,3,1), dt.datetime(2017,3,31)) # if G_Video33_1month is selected
        # # di.reduce_timeframe(dt.datetime(2017, 3, 1), dt.datetime(2017, 3, 7)) # if G_Video33_1week is selected
        #
        # # --- Remove inactive users (the ones with small number of sessions in total)
        # # di.remove_inactive_users(n_sessions=MIN_N_SESSIONS)
        # #
        # # ---------- Add categories -----------------------------
        # print(f'{datetime.datetime.now()} Import Categories')
        # di.import_categories_data(CAT_DATA_PATH)
        # print(f'{datetime.datetime.now()} Import Categories End')
        #
        # print(f'{datetime.datetime.now()} Filter Short Session')
        # # ---- Leave only sessions with at least specified number of articles
        # di.filter_short_sessions(n_items=min_items_n)
        # print(f'{datetime.datetime.now()} Filter Short Session End')
        #
        #
        # # ------ Create a graph on the base of the dataframe ----
        # print(f'{datetime.datetime.now()} Graph Manipulation')
        # gm = GraphManipulation(G_structure='USAC')
        # print(f'{datetime.datetime.now()} Graph Manipulation End')
        #
        # print(f'{datetime.datetime.now()} Create Graph')
        # gm.create_graph(di.user_ses_df)
        # print(f'{datetime.datetime.now()} Create Graph End')
        #
        # # Filter again, because dataframe filtering leaves sessions where the same article is repeatedly read several times
        # # gm.filter_sessions(gm.G, n_items=MIN_ITEMS_N)
        # # gm.filter_users(gm.G, n_sessions=MIN_N_SESSIONS)
        #
        # # ---------- Add locations ------------------------------
        # di.import_locations_data(LOC_DATA_PATH)
        # gm.add_locations_data(di.locations_data)
        # G = gm.G
        gm = GraphManipulation()
        G = nx.read_gpickle(f'./Data/{data_path}.gpickle')
        possible_subgraphs = [('U','A'),('S','A'),('A','C'),('A','L'), ('U','S','A'),('U','A','C'),('U','A','L'),('A','C','L'), ('S','A','C'), ('S','A','L'), ('U','S','A','C'), ('U','S','A','L'), ('U','A','C','L'), ('S','A','C','L'), ('U','S','A','C','L')]
        subgraph_movies = {('U','A'):(('U','M')),('S','A'):('S','M'),('A','C'):('M','C'),('A','L'):('M','L'),('U','S','A'):('U','S','M'),('U','A','C'):('U','M','C'),('U','A','L'):('U','M','L'),('A','C','L'):('M','C','L'), ('S','A','C'):('S','M','C'), ('S','A','L'):('S','M','L'), ('U','S','A','C'):('U','S','M','C'), ('U','S','A','L'):('U','S','M','L'), ('U','A','C','L'):('U','M','C','L'), ('S','A','C','L'):('S','M','C','L'), ('U','S','A','C','L'):('U','S','M','C','L')}

        subgraphs =[]
        for length in range(2,len(nodes)+1):
            x = list(itertools.permutations(nodes,length))
            for item in x:
                if item in possible_subgraphs:
                    subgraphs.append(item)
        # gm.filter_users(gm.G, n_sessions=min)
        gm.G = G
        gm.filter_sessions(gm.G, n_items=min_items_n)


        e = f'{n_clicks}'
        e+=(f'\nGENERAL STATISTICS')
        e+=(f'\nNumber of users:{len(gm.get_users(G))}')
        e +=(f'\nNumber of sessions:{len(gm.get_sessions(G))}')
        e +=(f'\nNumber of articles:{len(gm.get_articles(G))}')
        e +=(f'\nNumber of categories:{len(gm.get_categories(G))}')
        e +=(f'\nNumber of locations:{len(gm.get_locations(G))}')

        art_per_session = gm.get_articles_per_session(gm.G)
        e +=(f'\nAvg number of articles per session:{round(np.mean(art_per_session), 2)}')
        e +=(f'\nMax number of articles per session:{round(np.max(art_per_session), 2)}')

        ses_per_user = gm.get_sessions_per_user(gm.G)
        e +=(f'\nAvg number of sessions per user:{round(np.mean(ses_per_user), 2)}')
        e +=(f'\nMax number of sessions per user:{round(np.max(ses_per_user), 2)} ')

        tas = TimeAwareSplits(G)
        tas.create_time_split_graphs(G, num_splits=number_splits)
        # tas.create_time_window_graphs(short_days)
        tas.create_time_window_graphs(short_days)
        _dump_process = True
        short_back_timedelta = datetime.timedelta(days=short_days)
        e +=(f'\n\nTime span list:\n')
        counter = 0
        for timespan in tas.time_span_list:

            e+=(f'{timespan}\n')
            counter+=1
            if counter>1:
                counter=0
        pop = PopularityBasedRec(G, number_recommendations)

        # RWR_SA = PersonalizedPageRankBasedRec(number_recommendations)

        ae = AccuracyEvaluation(G)

        train_set_len = []
        train_len_dict = defaultdict(list)
        n_articles_train = []
        n_recommendation = dict()
        sessions_per_user_in_short_term = []
        avg_ses_len = defaultdict(list)

        for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

            e +=(f'\n\n======= Time split{tw_i} =======')

            n_recommendation[tw_i] = 0
            n_recommendation[f'{tw_i}_correct'] = 0
            # long_train_g = tw_iter[0]
            tw_iter[1].frozen = False
            test_g = tw_iter[1].copy()

            # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
            test_g = gm.filter_sessions(test_g, n_items=min_items_n)
            if len(test_g) == 0:
                continue

            # ------ 1. Create a time-ordered list of user sessions
            test_sessions = sorted(
                [(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'],
                key=lambda x: x[1])

            sessions_knn_dict = defaultdict(tuple)

            # For each step a ranked list of N recommendations is created
            for (s, s_datetime) in test_sessions:

                user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]
                if user=='U50000':
                    print(1)
                test_session_G = nx.Graph()
                test_session_G.add_node(user, entity='U')
                test_session_G.add_node(s, entity='S')
                test_session_G.add_edge(user, s, edge_type='US')

                # -----------------------------------------------------
                articles = sorted(
                    [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                    key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

                avg_ses_len[tw_i].append(len(articles))

                # print('----------\narticles:', articles)
                # print('session:', s, s_datetime)

                for i in range(min_items_n, len(articles)):
                    methods_to_be_evaluated = []
                    methos_to_be_evaluated_explainable = []
                    # ------------ Short-term training set ----
                    short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta,
                                                                    test_session_graph=test_session_G)
                    if len(short_train_g) == 0:
                        continue



                    test_session_G.add_nodes_from(articles[:i], entity='A')
                    for a in articles[:i]:
                        test_session_G.add_edge(s, a, edge_type='SA')
                        test_session_G.add_node(gm.map_category(a), entity='C')
                        test_session_G.add_edge(a, gm.map_category(a), edge_type='AC')
                        test_session_G.add_node(gm.map_location(a), entity='L')
                        test_session_G.add_edge(a, gm.map_location(a), edge_type='AL')

                    # ------------ Short Training Set (containing currently analyzed session!) ---------


                    # ----------- Long-term user training set ---
                    users_from_short_train = gm.get_users(short_train_g)
                    user_long_train_g = tas.create_long_term_user_train_set(user, s, s_datetime, articles[:i],
                                                                            users_from_short_train)
                    if len(user_long_train_g) == 0:
                        continue

                    train_set_len.append(len(gm.get_sessions(short_train_g)))
                    train_len_dict[tw_i].append(len(gm.get_sessions(short_train_g)))
                    n_articles_train.append(len(gm.get_articles(short_train_g)))
                    ses_per_user = gm.get_sessions_per_user(short_train_g)
                    sessions_per_user_in_short_term.append(Counter(ses_per_user))
                    subgraphs_train = []
                    # --- Create train graphs
                    for subgraph in subgraphs:
                        if len(subgraph)==2:
                            subgraphs_train.append((gm.create_subgraph_of_adjacent_entities(short_train_g,
                                                                    list_of_entities=[subgraph[0][0], subgraph[1]]),f'{subgraph[0]}_{subgraph[1]}'))
                        elif len(subgraph)==3:
                            subgraphs_train.append((gm.create_subgraph_of_adjacent_entities(short_train_g,
                                                                                           list_of_entities=[
                                                                                               subgraph[0],
                                                                                               subgraph[1],subgraph[2]]),f'{subgraph[0]}_{subgraph[1]}_{subgraph[2]}'))
                        elif len(subgraph)==4:
                            subgraphs_train.append((gm.create_subgraph_of_adjacent_entities(short_train_g,
                                                                                           list_of_entities=[
                                                                                               subgraph[0],
                                                                                               subgraph[1],
                                                                                               subgraph[2],subgraph[3]]),f'{subgraph[0]}_{subgraph[1]}_{subgraph[2]}_{subgraph[3]}'))
                        else:
                            subgraphs_train.append((gm.create_subgraph_of_adjacent_entities(short_train_g,
                                                                                           list_of_entities=[
                                                                                               subgraph[0],
                                                                                               subgraph[1],
                                                                                               subgraph[2],
                                                                                               subgraph[3],subgraph[4]]),f'{subgraph[0]}_{subgraph[1]}_{subgraph[2]}_{subgraph[3]}_{subgraph[4]}'))
                    # sa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A'])
                    # usa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A'])
                    # sac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C'])
                    # sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g,
                    #                                                       list_of_entities=['S', 'A', 'L'])
                    # usac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'C'])
                    # usal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'L'])



                    # -------------------------------------------------------------------------------
                    # --------------- SIMILARITIES --------------------------------------------------


                    # -----------------------------------------------------


                    # -----------------------------------------------------
                    # ------------------- SimRank -------------------------



                    # -----------------------------------------------------
                    # ------------------- RWR -----------------------------
                    # --- Run models

                    # RWR_SA.compute_transition_matrix(sa_train_g)
                    # RWR_USA.compute_transition_matrix(usa_train_g)
                    # RWR_SAC.compute_transition_matrix(sac_train_g)
                    # RWR_SAL.compute_transition_matrix(sal_train_g)
                    # RWR_USAC.compute_transition_matrix(usac_train_g)
                    # RWR_USAL.compute_transition_matrix(usal_train_g)
                    # RWR_USACL.compute_transition_matrix(short_train_g)

                    # --- Extract SS matrices
                    # RWR_SA.create_sessionsession_matrix()
                    # RWR_SA.create_sessionitem_matrix()

                    # RWR_USA.create_sessionsession_matrix()
                    # RWR_USA.create_sessionitem_matrix()
                    # RWR_USA.create_itemitem_matrix()
                    # RWR_SAC.create_sessionsession_matrix()
                    # RWR_SAC.create_sessionitem_matrix()
                    # RWR_SAC.create_itemitem_matrix()
                    # RWR_SAL.create_sessionsession_matrix()
                    # RWR_SAL.create_sessionitem_matrix()
                    # RWR_SAL.create_itemitem_matrix()
                    # RWR_USAC.create_sessionsession_matrix()
                    # RWR_USAC.create_sessionitem_matrix()
                    # RWR_USAC.create_itemitem_matrix()
                    # RWR_USAL.create_sessionsession_matrix()
                    # RWR_USAL.create_sessionitem_matrix()
                    # RWR_USAL.create_itemitem_matrix()
                    # RWR_USACL.create_sessionsession_matrix()
                    # RWR_USACL.create_sessionitem_matrix()
                    # RWR_USACL.create_itemitem_matrix()

                    # -----------------------------------------------------
                    # ------------------ PathSim --------------------------

                    # PathSim_AUA.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
                    # PathSim_ASA.compute_similarity_matrix(short_train_g, 'A', 'S', 1)
                    # PathSim_ACA.compute_similarity_matrix(short_train_g, 'A', 'C', 1)
                    # PathSim_ALA.compute_similarity_matrix(short_train_g, 'A', 'L', 1)

                    # -----------------------------------------------------
                    # ------------------ PathCount --------------------------

                    # PathCount_AUA.compute_similarity_matrix_my(short_train_g, 'A', 'U', 2)
                    # PathCount_ASA.compute_similarity_matrix_my(short_train_g, 'A', 'S', 1)
                    # PathCount_ACA.compute_similarity_matrix_my(short_train_g, 'A', 'C', 1)
                    # PathCount_ALA.compute_similarity_matrix_my(short_train_g, 'A', 'L', 1)


                    # -----------------------------------------------------
                    # ------------------- S-S PathSim ---------------------
                    # SKNN_PathSim_SAS.compute_similarity_matrix(short_train_g, 'S', 'A', 1)
                    # SKNN_PathSim_SACAS.compute_similarity_matrix(sac_train_g, 'S', 'C', 2)
                    # SKNN_PathSim_SALAS.compute_similarity_matrix(sal_train_g, 'S', 'L', 2)


                    # -----------------------------------------------------
                    # ------------------- S-S PathCounts ------------------
                    # SKNN_PathCount_SAS.compute_similarity_matrix_my(short_train_g, 'S', 'A', 1)
                    # SKNN_PathCount_SACAS.compute_similarity_matrix_my(sac_train_g, 'S', 'C', 2)
                    # SKNN_PathCount_SALAS.compute_similarity_matrix_my(sal_train_g, 'S', 'L', 2)


                    # -----------------------------------------------------
                    # ------------------- PathCounts for expl -------------

                    # PathCount_ASA.compute_similarity_matrix(short_train_g, 'A', 'S', 1)
                    # PathCount_ACA.compute_similarity_matrix(short_train_g, 'A', 'C', 1)
                    # PathCount_ALA.compute_similarity_matrix(short_train_g, 'A', 'L', 1)
                    # PathCount_AUA.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
                    #
                    # PathCount_SAS.compute_similarity_matrix(short_train_g, 'S', 'A', 1)
                    # PathCount_SCS.compute_similarity_matrix(short_train_g, 'S', 'C', 2)
                    # PathCount_SLS.compute_similarity_matrix(short_train_g, 'S', 'L', 2)
                    #
                    # PathCount_UAU.compute_similarity_matrix(user_long_train_g, 'U', 'A', 2)
                    # PathCount_UCU.compute_similarity_matrix(user_long_train_g, 'U', 'C', 3)
                    # PathCount_ULU.compute_similarity_matrix(user_long_train_g, 'U', 'L', 3)

                    # -------------------------------------------------------------------------------
                    # --------------- RECOMMENDATIONS -----------------------------------------------

                    session_categories = [gm.map_category(a) for a in articles[:i]]
                    session_timeviews = [gm.map_timeview(test_g, s, a) for a in articles[:i]]

                    # ------- POP --------------------------
                    # ------------------- Popularity ----------------------
                    if 'POP' in methods:
                        pop.compute_pop(short_train_g)
                        pop_rec = pop.predict_next(user, articles[:i])

                        # if len(pop_rec) == 0:
                        #     continue
                        # else:
                        methods_to_be_evaluated.append((pop_rec, 'POP'))

                    # ------- SimRank ----------------------
                    if 'Simrank' in methods:
                        simrank_models = []
                        for subgraph_train in subgraphs_train:
                            simrank = SimRankRec(number_recommendations)
                            simrank.compute_similarity_matrix(subgraph_train[0], max_iter=10)
                            simrank_models.append((simrank,subgraph_train[1]))
                        for simrank_model in simrank_models:
                            recommendation = simrank_model[0].predict_next(user, articles[:i], method=2)
                            if len(recommendation) == 0:
                                continue
                            else:
                                methods_to_be_evaluated.append((recommendation, f'Simrank_{simrank_model[1]}'))

                    # ------- RWR --------------------------
                    if 'RWR' in methods:
                        rwr_models = []
                        for subgraph_train in subgraphs_train:
                            RWR = PersonalizedPageRankBasedRec(number_recommendations)
                            RWR.compute_transition_matrix(subgraph_train[0])
                            RWR.create_itemitem_matrix()
                            rwr_models.append((RWR, subgraph_train[1]))
                        for rwr_model in rwr_models:
                            recommendation = rwr_model[0].predict_next(user,articles[:i])
                            if len(recommendation) == 0:
                                continue
                            else:
                                methods_to_be_evaluated.append((recommendation,f'RWR_{rwr_model[1]}'))
                    metapaths_a = []
                    if 'Pathsim' in methods:
                        pathsim_models = []
                        for subgraph_train in subgraphs_train:
                            pathsim = PathSimRec(number_recommendations)
                            nodes = subgraph_train[1].split('_')
                            for node in nodes:
                                if node !='A':
                                    if f'A_{node}_A' not in metapaths_a:
                                        pathsim.compute_similarity_matrix(short_train_g,'A',node,1)
                                        pathsim_models.append((pathsim,f'Pathsim_A_{node}_A'))
                                        metapaths_a.append(f'A_{node}_A')
                            if ('U' in nodes) and ('S' in nodes) and ('A' in nodes):
                                pathsim.compute_similarity_matrix(short_train_g,'A','U',2)
                                pathsim_models.append((pathsim,f'Pathsim_A_S_U_S_A'))
                        for pathsim_model in pathsim_models:
                                recommendation = pathsim_model[0].predict_next(user, articles[:i], method=2)
                                if len(recommendation)==0:
                                    continue
                                else:
                                    methods_to_be_evaluated.append((recommendation,pathsim_model[1]))

                    if 'PathCount' in methods:
                        pathcounts = []
                        pathcount_models = []
                        ab_rec = []
                        sb_rec = []
                        if 'A' in nodes:
                            # for path_len in range(1,len(nodes)-1):
                            for node in nodes:
                                if node != 'A':
                                    if f'A_{node}_A' not in pathcounts:
                                        pathcounts.append(f'A_{node}_A')
                                        pathsim = PathSimRec(number_recommendations)
                                        pathsim.compute_similarity_matrix(short_train_g,'A',node,1)
                                        pathcount_rec_dict =pathsim.predict_next_by_AB(articles[:i], option='ib',topN=False)
                                        pathcount_models.append((pathsim,f'PathCount_A{node}A',list(pathcount_rec_dict.keys())[:number_recommendations],pathcount_rec_dict))
                                        ab_rec.append(list(pathcount_rec_dict.keys())[:number_recommendations])
                            if ('U' in nodes) and ('S' in nodes) and ('A' in nodes):
                                pathcounts.append(f'A_S_U_S_A')
                                pathsim = PathSimRec(number_recommendations)
                                pathsim.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
                                pathcount_rec_dict = pathsim.predict_next_by_AB(articles[:i], option='ib', topN=False)
                                pathcount_models.append((pathsim, f'PathCount_A_S_U_S_A',
                                                         list(pathcount_rec_dict.keys())[:number_recommendations],
                                                         pathcount_rec_dict))
                                ab_rec.append(list(pathcount_rec_dict.keys())[:number_recommendations])



                             #Combine Recs
                            rec_ab_df = pd.DataFrame(index=set(x for l in ab_rec for x in l), columns=pathcounts)
                            for a in rec_ab_df.index:
                                for pathcount in pathcount_models:
                                    if len(pathcount[2])>0:

                                        rec_ab_df.loc[a, f'{pathcount[1][10:]}'] = pathcount[3][a] if a in list(pathcount[3].keys()) else 0
                            rec_ab_df = rec_ab_df.fillna(0)
                            for pathcount in pathcount_models:
                                recommendation = pathcount[2]
                                if len(recommendation) == 0:
                                    continue
                                else:
                                    methos_to_be_evaluated_explainable.append((recommendation, f'{pathcount[1]}',rec_ab_df))


                    # if any(len(m) == 0 for m in methods_to_be_evaluated):
                    #     continue

                    n_recommendation[tw_i] += 1

                    # ------- Measuring accuracy ----------------------
                    # ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP', s=s)

                    # ae.evaluate_recommendation(rec=simrank_sal_s_rec, truth=articles[i], method='SimRank_SAL(s)', s=s)
                    e += f'\n\nuser:{user}'
                    active_users = gm.get_users(short_train_g)
                    # e += f'\nactive users : {user in active_users}'
                    e += f'\nNext Article : {articles[i]}'
                    for method in methods_to_be_evaluated:
                        rec_counter = 0
                        ae.evaluate_recommendation(rec=method[0],truth=articles[i],method=method[1],s=s)
                        e += f'\n{method[1]}_rec: ['
                        for rec in method[0]:
                            rec_counter+=1
                            if rec == articles[i]:
                                e+= f'**{rec}**'
                                n_recommendation[f'{tw_i}_correct'] +=1
                            else:
                                e+=f'{rec}'
                            if rec_counter<len(method[0]):
                                e+=', '
                        e+= ']'
                    for method in methos_to_be_evaluated_explainable:
                        rec_counter = 0
                        ae.evaluate_recommendation(rec=method[0], truth=articles[i], method=method[1], s=s)
                        e += f'\n{method[1]}_rec: ['
                        for rec in method[0]:
                            rec_counter += 1
                            if rec == articles[i]:
                                e += f'**{rec}**'
                                n_recommendation[f'{tw_i}_correct'] += 1
                            else:
                                e += f'{rec}'
                            e += ' explained by '
                            for index in rec_ab_df.columns:
                                if rec_ab_df[index][rec] > 0:
                                    e += f'{index}: {rec_ab_df[index][rec]} '
                            if rec_counter < len(method[0]):
                                e += ', '
                        e += '] '

                        rec_ab_df = method[2]


            ae.evaluate_session()

            ae.evaluate_tw()
            # print('- Number of recommendations made:', n_recommendations)

        ae.evaluate_total_performance()

        avg_n_ses_per_train_per_period = [round(np.mean(l)) for l in train_len_dict.values()]
        avg_ses_len_per_period = [round(np.mean(l), 2) for l in avg_ses_len.values()]

        # e +=(f'\n\n\nNumber of sessions per user per short train period:\n{sessions_per_user_in_short_term}')
        e +=(f'\nNumber of recommendations per time split:{n_recommendation.values()}')
        e +=(f'\nTotal # of recs:{sum(n_recommendation.values())}')
        e +=(f'\nAverage # sessions per train per period {avg_n_ses_per_train_per_period}')
        e +=(f'\nAverage # artiles per session per period {avg_ses_len_per_period}')
        e +=(f'\nAverage # sessions in train:{round(np.mean(train_set_len), 2)}')
        e +=(f'\nAverage # articles in train:{round(np.mean(n_articles_train), 2)}')

        e+=('\n---------- METHODS EVALUATION -------------')

        methods = [k for k, v in sorted(ae.precision.items(), key=itemgetter(1), reverse=True)]
        for m in methods:
            e +=(f'\n--- {m}: Precision:{ae.precision[m]}, NDCG:{ae.ndcg[m]}, ILD:{ae.diversity[m]},Explainability:{ae.explainability[m]}')

        # exit()

        # --- Create period index for plotting
        p_start = tas.time_span_list[1][0]
        p_end = tas.time_span_list[len(tas.time_span_list) - 1][1] + datetime.timedelta(days=1)
        month_range = pd.date_range(p_start, p_end, freq='M')
        p = []
        for period in tas.time_span_list:
            p.append(datetime.datetime.strftime(period[1], format='%Y-%m-%d'))
        traces =[]
        for method in methods:
            trace = go.Scatter(x = p,y=ae.tw_precision[method],name=method)
            traces.append(trace)

        return {
            'data':traces,
            'layout': go.Layout(
            xaxis={
                      'title': 'Time_Period',

                  },
                  yaxis = {
                              'title': 'Precision',

                          })

        }

if __name__ == '__main__':
    app.run_server(debug= False,threaded = True)
