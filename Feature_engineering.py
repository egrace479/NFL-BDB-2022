#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


#small helper functions

def get_game_season(game_id, games):
    return games[games['gameId']==game_id]['season'].values[0]


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
    play = game[game['playId'] == play_id]
    
    return play


# In[ ]:


def get_event(game_id, play_id, tracking_f, event):
    '''
    This function creates a small dataframe for football tracking around the event.

    Parameters:
    -----------
    game_id, play_id - game and play of interest
    tracking_f - football-specific tracking dataframe the that game and play are in
    event - string of the event that we want to find, i.e., 'extra_point_attempt'
    ...

    Returns:
    -----------
    event_df - 11-row dataframe of tracking data around the event
    frame_id - frameId of event
    '''
    play = get_play(game_id, play_id, tracking_f)
    
    index = play.index[play['event']== event].values[0]
    event_df = play.loc[index-5:index+5,:]
    max_index = event_df['s'].idxmax()
    frame_id = play.loc[max_index]['frameId']
    
    return event_df, frame_id


# In[ ]:


def x_within_fg_bounds(x):
    ''' 
    Check if ball is crossing the fieldgoal line

    Parameters:
    -----------
    x - x-position of football

    '''
    return ((x>119) & (x<121)) | ((x>-1) & (x<1))

def compute_endzone_y_pos(game_id, play_id, tracking):
    ''' 
    Compute y-position of ball as it crosses the fieldgoal line in extrapoint or fieldgoal.

    Parameters:
    -----------
    game_id - gameId of play
    paly_id - playId of play
    tracking - The relevant tracking data for the given game/play

    Returns:
    --------
    mean_y - Mean y-value of ball given initial and final positions within fieldgoal x-boundaries
    '''

    play_example = get_play(game_id, play_id, tracking)

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
    track_fp - football tracking dataframe
    
    Returns:
    --------
    play_df - Play data with computed endzone y-position column

    '''
    play_df['endzone_y'] = play_df.index.map(
        lambda x: compute_endzone_y_pos(
            play_df.loc[x]['gameId'],
            play_df.loc[x]['playId'],
            track_fp
        )
    )

    return play_df


# In[ ]:


def find_kickline(event_df, frame_id):  
    '''
    This function gives a straightline expectation of where the football crosses the endzone.

    Parameters:
    -----------
    event_df - dataframe of tracking data around the event (from get_event)
    frame_id - frameId where the event occurs
    ...
    finds event_index from frame_id of play and uses the x & y values from event_index
    and event_index+1 to calculate a straight line trajectory of the football

    Returns:
    -----------
    y value expectation of football at x=120
    '''
    event_index = event_df.index[event_df['frameId']==frame_id].values[0]
    
    x1 = event_df['x'][event_index]
    y1 = event_df['y'][event_index]
    x2 = event_df['x'][event_index+1]
    y2 = event_df['y'][event_index+1]
    
    m = (y2-y1)/(x2-x1)
    
    return m*(120-x1)+y1

