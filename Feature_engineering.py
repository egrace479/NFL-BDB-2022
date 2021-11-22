#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 


# In[2]:


#small helper functions

def get_game_season(game_id, games):
    return games[games['gameId']==game_id]['season'].values[0]


# In[3]:


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


# In[4]:


def get_event(game_id, play_id, track_fp, event):
    '''
    This function creates a small dataframe for football tracking around the event.

    Parameters:
    -----------
    game_id, play_id - game and play of interest
    track_fp - football-specific tracking dataframe for play type
    event - string of the event that we want to find, i.e., 'extra_point_attempt'
    ...

    Returns:
    -----------
    event_df - 11-row dataframe of tracking data around the event
    event_index - the index of the moment of the kick based on max velocity
    #frame_id - frameId of event
    '''
    play_ex = get_play(game_id, play_id, track_fp)
    
    index = play_ex.index[play['event']== event].values[0]
    event_df = play_ex.loc[index-5:index+5,:]
    event_index = event_df['s'].idxmax()
    
    if event_index == event_df.index[-1]:
        event_df = play_ex.loc[index-10:index+10,:]
        
    #frame_id = play_ex.loc[event_index]['frameId']
    
    return event_df, event_index


# In[5]:


def x_within_fg_bounds(x):
    ''' 
    Check if ball is crossing the fieldgoal line

    Parameters:
    -----------
    x - x-position of football

    '''
    return ((x>118) & (x<122)) | ((x>-2) & (x<2))

def compute_endzone_y_pos(game_id, play_id, track_fp):
    ''' 
    Compute y-position of ball as it crosses the fieldgoal line in extrapoint or fieldgoal.

    Parameters:
    -----------
    game_id - gameId of play
    paly_id - playId of play
    track_fp - football-specific tracking dataframe for play type

    Returns:
    --------
    mean_y - Mean y-value of ball given initial and final positions within fieldgoal x-boundaries
    '''

    play_example = get_play(game_id, play_id, track_fp)

    try:

        '''
        Our data is low-resolution compared to the speed of the ball, so we check if the ball is within
        two yards of the fieldgoal, and compute the average position of the ball before and after it
        crosses the fieldgoal.
        '''

        within_bounds = play_example[x_within_fg_bounds(play_example['x'])]

        id1 = len(within_bounds) // 2 - 1
        id2 = (len(within_bounds) // 2) if (len(within_bounds) % 2 == 0) else (len(within_bounds) // 2 - 1)

        first_y = within_bounds.iloc[id1]['y']
        last_y = within_bounds.iloc[id2]['y']

        mean_y = (first_y + last_y) / 2
    
    except:
        '''
        If there is no data returned from the above computation, fill with NaN
        '''
        mean_y = np.nan

    return mean_y

def endzone_y_pos(play_df, track_fp):
    
    ''' 
    Compute y-position of ball as it crosses fieldgoal line for each play (extra point or fieldgoal).

    Paramters:
    ----------
    play_df - play dataframe for desired play type
    track_fp - football tracking dataframe for desired play type
    
    Returns:
    --------
    play_df - play dataframe for desired play type with endzone y-position column

    '''
    play_df['endzone_y'] = play_df.index.map(
        lambda x: compute_endzone_y_pos(
            play_df.loc[x]['gameId'],
            play_df.loc[x]['playId'],
            track_fp
        )
    )

    return play_df


# In[6]:


def find_kickline(game_id, play_id, track_fp, event):
    
    '''
    This function gives a straightline expectation of where the football crosses the endzone.

    Parameters:
    -----------
    game_id, play_id - game and play of interest
    track_fp - football-specific tracking dataframe for play type
    event - string of the event that we want to find, i.e., 'extra_point_attempt'
    ...
    finds event_index from frame_id of play and uses the x & y values from event_index
    and event_index+1 to calculate a straight line trajectory of the football

    Returns:
    -----------
    y value expectation of football at x=120
    '''
    event_df, event_index = get_event(game_id, play_id, track_fp, event)
    #event_index = event_df.index[event_df['frameId']==frame_id].values[0]
    
    x1 = event_df['x'][event_index]
    y1 = event_df['y'][event_index]
    x2 = event_df['x'][event_index+1]
    y2 = event_df['y'][event_index+1]
    
    m = (y2-y1)/(x2-x1)
    
    return m*(120-x1)+y1


# In[7]:


def expected_endzone_y_pos(pt_play, track_fp, event):
    ''' 
    The expected y-position of ball as it crosses fieldgoal line for each play (extra point or fieldgoal) based on a straight 
    line estimate.

    Paramters:
    ----------
    pt_play - play dataframe for desired play type
    track_fp - football tracking dataframe for desired play type
    event - string of the event that we want to find, i.e., 'extra_point_attempt'
    
    Returns:
    --------
    pt_play - play dataframe for desired play type with computed endzone y-position column

    '''
    pt_play['expected_endzone_y'] = pt_play.index.map(
        lambda x: find_kickline(
            pt_play.loc[x]['gameId'],
            pt_play.loc[x]['playId'],
            track_fp, event
        )
    )

    return pt_play


# In[ ]:


def exp_err_y(pt_play):
    ''' 
    The difference between the expected y-position of ball as it crosses fieldgoal line and the actual y-position
    for each play (extra point or fieldgoal).

    Paramters:
    ----------
    pt_play - expanded play dataframe for desired play type (includes cols: 'expected_endzone_y' and 'endzone_y')
    
    Returns:
    --------
    pt_play - play dataframe for desired play type with computed endzone y-position error column

    '''
    pt_play['endzone_y_error'] = np.abs(pt_play['endzone_y'] - pt_play['expected_endzone_y'])
    
    return pt_play


# In[ ]:


def off_center(pt_play):
    ''' 
    The difference between the y-position of ball as it crosses fieldgoal line and the center of the fieldgoal
    for each play (extra point or fieldgoal).

    Paramters:
    ----------
    pt_play - expanded play dataframe for desired play type (includes cols: 'expected_endzone_y' and 'endzone_y')
    
    Returns:
    --------
    pt_play - play dataframe for desired play type with computed endzone y-position off-from-center column

    '''
    pt_play['endzone_y_off_center'] = np.abs(pt_play['endzone_y'] - (160/6))
    
    return pt_play


# In[8]:


def l2_norm(x1, y1, x2, y2):
    # Computes euclidean distance between two points
    return np.sqrt(np.square(x1-x2) + np.square(y1-y2))


# In[9]:


def get_opposing_team(kicking_team):
    # Returns label of opposing team
    return 'home' if kicking_team == 'away' else 'away'


# In[10]:


def compute_kicker_core_dist(game_id, play_id, tracking, track_fp, k, event):
    '''
    Compute core distance from kicker to players on opposing team

    Parameters:
    -----------
    game_id - gameId of play
    play_id - playId of play
    tracking - Tracking data containing relevant play
    k - Number of nearest neighbors to check (returns distance of k-th nearest player)
    #we seem to need track_fp to get the event of the kick

    Returns:
    --------
    core_distance - The core distance from kicker to players on opposing team

    '''

    # Get play and event data
    play_ex = get_play(game_id, play_id, tracking)
    event_df, event_index = get_event(game_id, play_id, track_fp, event)

    
    # Get all play data at the time of the kick
    kick_frame = event_df.loc[event_index]['frameId']
    kick_tracking = play_ex[play_ex['frameId']==kick_frame]

    # Get data from players on opposing team
    try:
        kicking_team = kick_tracking[kick_tracking['position']=='K']['team'].values[0]
        
    except IndexError:

        '''
        Apparently there are cases in which a kicker isn't present (???)
        so if grabbing the kicker's team fails, just return null.
        '''

        return np.nan

    opposing_team = get_opposing_team(kicking_team)
    opposing_team_players = kick_tracking[kick_tracking['team']==opposing_team]

    # Get kicker x and y coords
    kicker_x = kick_tracking[kick_tracking['position']=='K']['x'].values[0]
    kicker_y = kick_tracking[kick_tracking['position']=='K']['y'].values[0]

    # Compute Euclidean distances from kicker to players on opposing team
    opposing_team_players['kicker_dist'] = l2_norm(kicker_x, kicker_y, opposing_team_players['x'], opposing_team_players['y'])

    # Sort distances, grab k-th nearest distance (the core distance)
    sorted_distances = opposing_team_players['kicker_dist'].sort_values()
    core_distance = sorted_distances.iloc[k-1]

    return core_distance
    


# In[11]:


def kicker_core_dist(pt_play, track_pt18, track_pt19, track_pt20, track_fp, event, k=5):
    '''
    Find core distance from kicker to players on opposing team. Wrapper function to call compute.

    Parameters:
    -----------
    pt_play - play dataframe for desired play type
    track_pt18, track_pt19, track_pt20 - tracking dataframes for play-type for each year
    track_fp - football tracking dataframe for desired play type
    event - string of the event that we want to find, i.e., 'extra_point_attempt'
    k - Number of nearest neighbors to check (returns distance of k-th nearest player)
    #we seem to need track_fp to get the event of the kick

    Returns:
    --------
    pt_play - play dataframe for desired play type with new column 'kicker_core_dist'

    '''
    
    tracking = pd.concat([track_pt18, track_pt19, track_pt20])

    pt_play['kicker_core_dist'] = pt_play.index.map(
        lambda x: compute_kicker_core_dist(
            pt_play.loc[x]['gameId'],
            pt_play.loc[x]['playId'],
            tracking,
            track_fp,
            event,
            k=k
        )
    )

    return pt_play


# In[ ]:




