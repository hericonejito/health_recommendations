import pandas as pd
import datetime as dt

class DataImport:

    def __init__(self):
        self.user_ses_df = None

    def import_decagon_data(self,data_path,adjust_pk_names=False,custom_user = 'U', order_by_time=True, pk_item = 'pk_article',custom_session = 'S',custom_article = 'A',seperator='\t'):
        user_ses_df = pd.read_csv(data_path, sep=seperator, header=None)
        print(user_ses_df)
        user_ses_df.columns = ['pk_user', 'pk_session', pk_item, ]


        if adjust_pk_names == True:
            user_ses_df['pk_user'] = custom_user + user_ses_df['pk_user'].astype('str')
            user_ses_df['pk_session'] = custom_session + user_ses_df['pk_session'].astype('str')
            user_ses_df[pk_item] = custom_article + user_ses_df[pk_item].astype('str')
        user_ses_df = user_ses_df.fillna(0)
        self.user_ses_df = user_ses_df
    def import_user_click_data(self, data_path, adjust_pk_names=False,custom_user = 'U', order_by_time=True, pk_item = 'pk_article',custom_session = 'S',custom_article = 'A',seperator='\t'):
        '''
        Import user click and session data from data_path and
        return it in a pandas Dataframe (ordered by time if requested)
        '''

        self.pk_item = pk_item

        user_ses_df = pd.read_csv(data_path, sep=seperator, header=None)
        print(user_ses_df)
        user_ses_df.columns = ['pk_user', 'pk_session', pk_item, 'timeview', 'date', 'time']
        user_ses_df['date-time'] = pd.to_datetime(user_ses_df['date'] + ' ' + user_ses_df['time'],
                                                  format='%Y-%m-%d %H:%M:%S')
        user_ses_df.drop(['date', 'time'], axis=1, inplace=True)
        if order_by_time:
            user_ses_df.sort_values('date-time', inplace=True)

        if adjust_pk_names == True:
            user_ses_df['pk_user'] = custom_user + user_ses_df['pk_user'].astype('str')
            user_ses_df['pk_session'] = custom_session + user_ses_df['pk_session'].astype('str')
            user_ses_df[pk_item] = custom_article + user_ses_df[pk_item].astype('str')
        user_ses_df = user_ses_df.fillna(0)
        self.user_ses_df = user_ses_df


    def filter_short_sessions(self, n_items=2):
        '''
        From the dataframe remove those sessions that have less than 2 articles
        '''
        articles_per_session = self.user_ses_df.groupby(['pk_session'])[self.pk_item].nunique()
        long_sessions = articles_per_session[articles_per_session >= n_items]

        self.user_ses_df = self.user_ses_df[self.user_ses_df['pk_session'].isin(long_sessions.keys())]


    def reduce_timeframe(self, from_date=dt.datetime(2015,1,1), to_date=dt.datetime(2018,12,31)):
        '''
        Reduce the dataset by filtering only the dates from the given period
        '''
        self.user_ses_df = self.user_ses_df[(self.user_ses_df['date-time'] >= from_date)
                                            & (self.user_ses_df['date-time'] <= to_date)]


    def import_categories_data(self, data_path):

        categories_data = pd.read_csv(data_path, sep='\t',
                                      names=['id', 'article', 'category1', 'category2', 'category3', 'category4', 'category5'])
        categories_data['article'] = 'A' + categories_data['article'].astype('str')
        categories_data.drop('id', 1, inplace=True)
        categories_data.set_index('article', inplace=True)

        # Create a dict from the dataframe
        cat_dict = categories_data.to_dict('index')

        # For each article assign the most probable category
        # (later can be assigned several with different weights)
        article_cat = dict()
        for a in cat_dict:
            most_probable_cat = [k for k, v in cat_dict[a].items() if v == max(cat_dict[a].values())][0]
            article_cat[a] = most_probable_cat

        cat_df = pd.DataFrame(list(article_cat.items()), columns=[self.pk_item, 'pk_category'])

        self.user_ses_df = pd.merge(self.user_ses_df, cat_df, how='right', on=[self.pk_item, self.pk_item])


    def import_video_categories(self, data_path):

        video_cat_data = pd.read_csv(data_path, sep='\t', names=[self.pk_item, 'video_category_id'])
        video_cat_data[self.pk_item] = 'A' + video_cat_data[self.pk_item].astype('str')
        video_cat_data['video_category_id'] = 'C' + video_cat_data['video_category_id'].astype('str')
        video_cat_data.set_index(self.pk_item, inplace=True)

        self.video_cat_data = video_cat_data


    def import_locations_data(self, data_path):

        locations_data = pd.read_csv(data_path, sep='\t', names=['article', 'location'])
        locations_data['article'] = 'A' + locations_data['article'].astype('str')

        self.locations_data = locations_data



    def remove_inactive_users(self, n_sessions=2):

        sessions_per_user = self.user_ses_df.groupby(['pk_user'])['pk_session'].nunique()
        active_users = sessions_per_user[sessions_per_user >= n_sessions]

        self.user_ses_df = self.user_ses_df[self.user_ses_df['pk_user'].isin(active_users.keys())]