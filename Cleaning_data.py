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


# In[5]:


def drop_by_index_difference(pt_play, track_fp, event, threshold):
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

    index_diff = pd.Series(index_diff, index=play_df.index)

    # Filter using the above series as a boolean mask
    filtered_pt_play = pt_play[index_diff <= threshold]

    return filtered_pt_play


# In[ ]:




