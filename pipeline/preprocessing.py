import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from querying.tracking_query import get_play

def ft_in(x):
    if '-' in x:
        meas=x.split('-')
        #this will be a list ['ft','in']
        inches = int(meas[0])*12 + int(meas[1])
        return inches
    else:
        return int(x)

def clock(x,df):
    gameClock = df.loc[x]['gameClock']
    quarter = df.loc[x]['quarter']

    gameClock_split = gameClock.split(':')

    minutes = gameClock_split[0]
    seconds = gameClock_split[1]

    total_minutes = 15-int(minutes) + 15 * (quarter - 1)

    return (total_minutes * 60) + int(seconds)

def get_game_season(game_id, games):
    return games[games['gameId']==game_id]['season'].values[0]

def preprocess_players(players_df):
    # preprocessing steps
    players_df['height'] = players_df['height'].apply(ft_in)
    return players_df

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

# #### Preprocessing functions for actual modeling or clustering.

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
                   'preSnapVisitorScore', 'kicker_height', 'kicker_weight', #'endzone_y_expected', 
                   'endzone_y', 'endzone_y_error','endzone_y_off_center']
    
    columns = ep_plays.columns
    
    useful_cols.extend(col for col in columns if 'kicker_core_dist' in col)
                
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
              'playResult', #'endzone_y_expected', 
                   'endzone_y', 'endzone_y_error','endzone_y_off_center']
    
    columns = fg_plays.columns
    
    useful_cols.extend(col for col in columns if 'kicker_core_dist' in col)
    
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

def get_kick_attempt_idx_diff(game_id, play_id, track_fp, event):
    '''
    For a given gameId and playId, return the difference in index between the event as labelled and the maximum speed of the ball.
    Note we only use the position of the football to see if the laws of physics are broken.  
    
    Parameters:
    -----------
    game_id - gameId of play
    play_id - playId of play
    track_fp - football-specific tracking dataframe for play type
    event - The event type to compute an index difference for

    Returns:
    --------
    idx_diff - The difference in index between the labelled event and the max speed of the ball

    '''
    # get play and football data, feed track_fp into get_play function to get only football information on the play.
    ball_ex = get_play(game_id, play_id, track_fp)

    # Return null if desired event not present
    if event not in ball_ex['event'].values:
        return np.nan

    # Compute event index and max ball speed index
    event_idx = ball_ex[ball_ex['event']==event].index[0]
    max_speed_idx = ball_ex['s'].idxmax()

    # Compute difference in indices
    idx_diff = np.abs(event_idx - max_speed_idx)

    return idx_diff

def drop_by_index_difference(pt_play, track_fp, event, threshold=7):
    '''
    Drop values from play DataFrame according to event-vs-max-speed index difference.

    Parameters:
    -----------
    pt_play - DataFrame containing data for a specific play type (e.g. field goals, extra points)
    track_fp - football-specific tracking dataframe for play type
    event - The event type to compute index difference for

    Returns:
    --------
    filtered_play_df - play_df with rows dropped according to index difference threshhold

    '''

    # Create a pandas Series of index differences
    index_diff = pt_play.index.map(
        lambda x: get_kick_attempt_idx_diff(
            pt_play.loc[x]['gameId'],
            pt_play.loc[x]['playId'],
            track_fp,
            event
        )
    )

    index_diff = pd.Series(index_diff, index=pt_play.index)

    # Filter using the above series as a boolean mask
    filtered_pt_play = pt_play[index_diff <= threshold]

    return filtered_pt_play