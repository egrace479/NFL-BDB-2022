#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hdbscan


# In[ ]:


##Helper Functions :)
#convert feet to inches for players
def ft_in(x):
    if '-' in x:
        meas=x.split('-')
        #this will be a list ['ft','in']
        inches = int(meas[0])*12 + int(meas[1])
        return inches
    else:
        return int(x)
    
#convert Game Clock from MM:SS:00 to Seconds
def clock(x,df):
    gameClock = df.loc[x]['gameClock']
    quarter = df.loc[x]['quarter']

    gameClock_split = gameClock.split(':')

    minutes = gameClock_split[0]
    seconds = gameClock_split[1]

    total_minutes = int(minutes) + 15 * (quarter - 1)

    return (total_minutes * 60) + int(seconds)


# In[ ]:


def get_play(game_id, play_id, tracking):
    '''
    This function creates the tracking dataframes.

    Parameters:
    -----------
    game_id, play_id - game and play of interest
    tracking - tracking dataframe the that game and play are in
    ...

    Returns:
    -----------
    play - dataframe of just the tracking data for the particular play of interest
    '''
    game = tracking[tracking['gameId'] == game_id]
    play_ex = game[game['playId'] == play_id]
    
    return play_ex


# In[ ]:


##preprocess by dataset
#players Dataset
def preprocess_players(players_df):
    # preprocessing steps
    players_df['height'] = players_df['height'].apply(ft_in)
    return players_df


# Note: run `preprocess_tracking` first, then `preprocess_football_track`, then `preprocess_play`.
# AFTER running these 3 functions, you can run `drop_by_index_difference`.

# In[ ]:


#preprocess tracking
def preprocess_tracking(track18, track19, track20, play_df, play_type):
    '''
    This function creates the tracking dataframes by play-type by year.

    Parameters:
    -----------
    track18, track19, track20 - trackYY.csv dataframes
    play_df - play.csv dataframe
    play_type - string, play type, e.g., 'Extra Point' 
    ...

    Returns:
    -----------
    track_p18 - Tracking Play Type 2018 dataframe
    track_p19 - Tracking Play Type 2019 dataframe
    track_p20 - Tracking Play Type 2020 dataframe
    '''
    track18 = track18.copy()
    track19 = track19.copy()
    track20 = track20.copy()
    #re-orient direction of play by offensive team direction : 
    #We must reorient this to reflect movement in the offense direction instead of the on-field coordinates 
    #(reorient the origin from the bottom left to top right for a change in direction).
    #2018 tracking data
    track18.loc[track18['playDirection'] == 'left', 'x'] = 120 -track18.loc[track18['playDirection']=='left','x']
    track18.loc[track18['playDirection'] == 'left', 'y'] = 160/3 -track18.loc[track18['playDirection']=='left','y']
    #note that we have 160/3 for the y direction since the football field is 160ft, but our units are yards

    #2019 tracking data
    track19.loc[track19['playDirection'] == 'left', 'x'] = 120 -track19.loc[track19['playDirection']=='left','x']
    track19.loc[track19['playDirection'] == 'left', 'y'] = 160/3 -track19.loc[track19['playDirection']=='left','y']

    #2020 tracking data
    track20.loc[track20['playDirection'] == 'left', 'x'] = 120 -track20.loc[track20['playDirection']=='left','x']
    track20.loc[track20['playDirection'] == 'left', 'y'] = 160/3 -track20.loc[track20['playDirection']=='left','y']

    #divide play dataset by type of play
    play_p = play_df.loc[play_df['specialTeamsPlayType']== play_type][['gameId', 'playId']]
    
    #merge play_type with tracking for each year
    track_p18 = pd.merge(play_p, track18, left_on = ['gameId', 'playId'], right_on = ['gameId', 'playId'])
    track_p19 = pd.merge(play_p, track19, left_on = ['gameId', 'playId'], right_on = ['gameId', 'playId'])
    track_p20 = pd.merge(play_p, track20, left_on = ['gameId', 'playId'], right_on = ['gameId', 'playId'])
    
    return track_p18, track_p19, track_p20


# In[ ]:


def preprocess_football_track(track_p18, track_p19, track_p20):
    '''
    This function creates the football tracking dataframe by given event.

    Parameters:
    -----------
    track_p18, track_p19, track_p20 - tracking by play by year dataframes
    ...

    Returns:
    -----------
    track_fp - Tracking Football Play Type dataframe
    '''
    
    #separate out the football data in each year's tracking dataframe and drop null values
    #concatenate to one dataframe for football tracking data
    track_fp18 = track_p18.loc[track_p18['displayName'] == 'football'].dropna(axis = 'columns')
    track_fp19 = track_p19.loc[track_p19['displayName'] == 'football'].dropna(axis = 'columns')
    track_fp20 = track_p20.loc[track_p20['displayName'] == 'football'].dropna(axis = 'columns')
    track_fp = pd.concat([track_fp18, track_fp19, track_fp20], ignore_index = True)
    
    return track_fp


# In[ ]:


#preprocess play data
def preprocess_play(play_df):
    '''
    This function fills nulls in the play dataframe and applies the clock function.

    Parameters:
    -----------
    play_df - play.csv dataframe
    ...

    Returns:
    -----------
    play_df - Processed play dataframe
    '''
    #null penalty yards = 0
    play_df['penaltyYards']=play_df['penaltyYards'].fillna(0)
    
    #clock: MM:SS to Seconds
    play_df['gameClockSeconds'] = play_df.index.map(lambda x: clock(x,play_df))
    
    #redefine nulls in penalty as no penalty
    play_df['penaltyCodes']=play_df['penaltyCodes'].fillna('no penalty')
    
    #TO-DO: Address null values on kickerId and kickBlockerId 
    #(note their height & weight comes in too)
    return play_df


# #### We need to make some data frames for analysis: Weather, ExtraPoint, FieldGoal, Punts, Kickoffs

# In[ ]:


def get_weather_data():
    '''
    This function creates the Weather dataframes by year.

    Returns:
    -----------
    weather2018, weather2019, weather2020 - Weather dataframes by year

    '''
    # Pull down datasets
    games = pd.read_csv('https://raw.githubusercontent.com/ThompsonJamesBliss/WeatherData/master/data/games.csv')
    stadium_coordinates = pd.read_csv('https://raw.githubusercontent.com/ThompsonJamesBliss/WeatherData/master/data/stadium_coordinates.csv')
    games_weather = pd.read_csv('https://raw.githubusercontent.com/ThompsonJamesBliss/WeatherData/master/data/games_weather.csv')

    # Merge game and weather data on game_id
    games_weather_merge = pd.merge(games_weather, games, on='game_id')

    # Merge stadium data on StadiumName
    final_df = pd.merge(games_weather_merge, stadium_coordinates, on='StadiumName')

    # Convert time columns to datetime objects
    time_cols = ['TimeMeasure', 'TimeStartGame', 'TimeEndGame']

    for col in time_cols:
        final_df[col] = pd.to_datetime(final_df[col], format='%m/%d/%Y %H:%M')

    # Create sliced DataFrames
    weather2018 = final_df[final_df['TimeMeasure'].dt.year == 2018]
    weather2019 = final_df[final_df['TimeMeasure'].dt.year == 2019]
    weather2020 = final_df[final_df['TimeMeasure'].dt.year == 2020]

    return weather2018, weather2019, weather2020


# In[ ]:


#make the ExtraPoint dataframe
#this runs AFTER play and players are preprocessed
def make_extraPoint(play_df, players_df):
    '''
    This function creates the ExtraPoint dataframe.

    Parameters:
    -----------
    play_df - Preprocessed players.csv dataframe
    ...

    Returns:
    -----------
    ep_plays - ExtraPoint dataframe

    '''
    play_extrapoint = play_df.loc[play_df['specialTeamsPlayType']=='Extra Point']
    #remove extraneous columns
    ep = play_extrapoint.drop(columns =['kickReturnYardage', 'kickLength', 'playResult', 'returnerId', 'yardsToGo', 'down', 'specialTeamsPlayType'])
   
    #add in Kickers
    ep_play = pd.merge(ep, players_df[['nflId', 'height', 'weight','Position', 'displayName']], how = 'left',
             left_on = 'kickerId', right_on = 'nflId')
    ep_plays=ep_play.rename(columns = {"height": 'kicker_height', "weight": 'kicker_weight', "Position": 'kicker_position', "displayName": 'kicker_name'})
    ep_plays=ep_plays.drop(columns=['nflId'])
    #add in Blockers (figure out Nulls first!)
    #ep_full = pd.merge(ep_plays, players_df[['nflId', 'height', 'weight','Position', 'displayName']], how = 'left',
    #         left_on = 'kickBlockerId', right_on = 'nflId')
    #eps=ep_full.rename(columns = {"height": 'blocker_height', "weight": 'blocker_weight', "Position": 'blocker_position', "displayName": 'blocker_name'})
    #eps=eps.drop(columns=['nflId'])
    return ep_plays
    


# In[ ]:


#make FieldGoal dataframe
def make_fieldGoal(play_df, players_df):
    '''
    This function creates the FieldGoal dataframe.

    Parameters:
    -----------
    play_df - Preprocessed players.csv dataframe
    ...

    Returns:
    -----------
    fg_plays - FieldGoal dataframe

    '''
    play_fieldgoal = play_df.loc[play_df['specialTeamsPlayType']=='Field Goal']
    #remove extraneous columns
    fg = play_fieldgoal.drop(columns =['kickReturnYardage', 'specialTeamsPlayType'])
   
    #add in Kickers
    fg_play = pd.merge(fg, players_df[['nflId', 'height', 'weight','Position', 'displayName']], how = 'left',
             left_on = 'kickerId', right_on = 'nflId')
    fg_plays=fg_play.rename(columns = {"height": 'kicker_height', "weight": 'kicker_weight', "Position": 'kicker_position', "displayName": 'kicker_name'})
    fg_plays=fg_plays.drop(columns=['nflId'])
    #add in Blockers (figure out Nulls first!)
    #fg_full = pd.merge(fg_plays, players_df[['nflId', 'height', 'weight','Position', 'displayName']], how = 'left',
    #         left_on = 'kickBlockerId', right_on = 'nflId')
    #fgs=fg_full.rename(columns = {"height": 'blocker_height', "weight": 'blocker_weight', "Position": 'blocker_position', "displayName": 'blocker_name'})
    #fgs=fgs.drop(columns=['nflId'])
    return fg_plays


# #### Preprocessing functions for actual modeling or clustering.

# In[ ]:


def preprocess_ep(ep_plays, encode_categorical = True):
    '''
    This function the ExtraPoint dataframe for clustering.

    Parameters:
    -----------
    ep_plays - ExtraPoint dataframe
    encode_categorical - Boolean, default is True
    ...

    Returns:
    -----------
    ep_scale - scaled/processed ExtraPoint dataframe
    ep_df - truncated ExtraPoint Dataframe without the scaling.

    '''
    #reduce number of columns to those with numeric values or one-hot-encode the categoricals
    useful_cols = ['specialTeamsResult', 'yardlineNumber', 'gameClockSeconds', 
                   'penaltyCodes', 'penaltyYards', 'preSnapHomeScore', 
                   'preSnapVisitorScore', 'kicker_height', 'kicker_weight', #'expected_endzone_y', 
                   'endzone_y', 'endzone_y_error','endzone_y_off_center']
    
    columns = ep_plays.columns
    
    useful_cols.extend(col for col in columns if 'kicker_core_dist' in col)
    
    #useful_cols with blockers
    #useful_cols = ['specialTeamsResult', 'yardlineNumber', 'gameClockSeconds', 
                 #   'penaltyCodes', 'penaltyYards', 'preSnapHomeScore', 'preSnapVisitorScore', 
                # 'kicker_height', 'kicker_weight', 'blocker_height', 'blocker_weight',
                # 'expected_endzone_y', 'endzone_y', 'kicker_core_dist']
                
    #need to drop nulls for clustering
    ep_df = ep_plays[useful_cols].dropna()
    
    new_eps = ep_df.drop(['specialTeamsResult', 'penaltyCodes'], axis=1)
    #new_eps['specialTeamsResult'] = ohe_str
    #new_eps['penaltyCodes'] = ohe_pc
    
    #scale data, but only non-categorical columns
    scale = StandardScaler()
    ep_scale = scale.fit_transform(new_eps)
    #TO-DO QUESTION: do we want to scale categoricals too? No
    
    #make this back into a data frame
    ep_scale = pd.DataFrame(ep_scale, columns = new_eps.columns)
    
    #add categorical columns back
    if encode_categorical:
        #one-hot-encode SpecialTeamsResult and penaltyCodes
        le_str = LabelEncoder()
        le_pc = LabelEncoder()
        ohe_str = le_str.fit_transform(ep_df['specialTeamsResult'])
        ohe_pc = le_pc.fit_transform(ep_df['penaltyCodes'])
        
        ep_scale['specialTeamsResult'] = ohe_str
        ep_scale['penaltyCodes'] = ohe_pc
    else:
        ep_scale['specialTeamsResult'] = ep_df['specialTeamsResult']
        ep_scale['penaltyCodes'] =ep_df['penaltyCodes']
        
    return ep_scale, ep_df
    


# In[ ]:


def preprocess_fg(fg_plays, encode_categorical=True):
    '''
    This function the FieldGoal dataframe for clustering.

    Parameters:
    -----------
    fg_plays - FieldGoal dataframe
    encode_categorical - Boolean, default is True
    ...

    Returns:
    -----------
    fg_scale - scaled/processed FieldGoal dataframe
    fg_df - truncated FieldGoal Dataframe without the scaling.

    '''
    #reduce number of columns to those with numeric values or one-hot-encode categoricals
    useful_cols = ['specialTeamsResult', 'yardlineNumber', 
               'gameClockSeconds', 'penaltyCodes', 
               'penaltyYards', 'preSnapHomeScore', 
               'preSnapVisitorScore', 'kicker_height', 
               'kicker_weight', 'down',
              'yardsToGo', 'kickLength',
              'playResult', #'expected_endzone_y', 
                   'endzone_y', 'endzone_y_error','endzone_y_off_center']
    
    columns = fg_plays.columns
    
    useful_cols.extend(col for col in columns if 'kicker_core_dist' in col)
    
    #useful_cols with blockers
    #useful_cols = ['specialTeamsResult', 'yardlineNumber', 
   #            'gameClockSeconds', 'penaltyCodes', 
   #            'penaltyYards', 'preSnapHomeScore', 
   #            'preSnapVisitorScore', 'kicker_height', 
   #            'kicker_weight', 'blocker_height', 
    #           'blocker_weight', 'down',
    #          'yardsToGo', 'kickLength',
    #          'playResult', 'expected_endzone_y', 
    #             'endzone_y', 'kicker_core_dist']
    
    #need to drop nulls for clustering
    fg_df = fg_plays[useful_cols].dropna()
    
    new_fgs = fg_df.drop(['specialTeamsResult', 'penaltyCodes'], axis=1)
    #new_fgs['specialTeamsResult'] = ohe_str
    #new_fgs['penaltyCodes'] = ohe_pc
    
    #scale data, but only non-categorical columns
    scale = StandardScaler()
    fg_scale = scale.fit_transform(new_fgs)
    
    #make this back into a data frame
    fg_scale = pd.DataFrame(fg_scale, columns = new_fgs.columns)
    
    #add categorical columns back
    if encode_categorical:
        #one-hot-encode SpecialTeamsResult and penaltyCodes
        le_str = LabelEncoder()
        le_pc = LabelEncoder()
        ohe_str = le_str.fit_transform(fg_df['specialTeamsResult'])
        ohe_pc = le_pc.fit_transform(fg_df['penaltyCodes'])
        
        fg_scale['specialTeamsResult'] = ohe_str
        fg_scale['penaltyCodes'] = ohe_pc
    else:
        fg_scale['specialTeamsResult'] = fg_df['specialTeamsResult']
        fg_scale['penaltyCodes'] = fg_df['penaltyCodes']
    
    return fg_scale, fg_df


# Let's make the clustering pipeline.

# In[ ]:


def cluster_df(df_scale, df):
    '''
    This function performs the clustering on the dataframe df through the scaled data
    
    Parameters:
    -----------
    df_scale - the array produced from StandardScaler on the entire preprocessed dataframe df
    df - the preprocessed dataframe, prior to encoding
    
    Returns:
    ---------
    df with column 'cluster_id' to track cluster labels
    cls - the fit cluster object to make trees, etc.
    '''
    clusterer = hdbscan.HDBSCAN()
    cls = clusterer.fit(df_scale)
    df['cluster_id']=cls.labels_
    
    return cls, df
    

