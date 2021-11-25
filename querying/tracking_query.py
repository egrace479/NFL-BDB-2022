def get_play(game_id, play_id, tracking):
    game = tracking[tracking['gameId'] == game_id]
    play = game[game['playId'] == play_id]
    return play

def get_play_frame(play, frame_id):
    frame = play[play['frameId'] == frame_id]

    home = frame[frame['team'] == 'home']
    away = frame[frame['team'] == 'away']
    ball = frame[frame['team'] == 'football']

    return home, away, ball

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
    
    index = play_ex.index[play_ex['event']== event].values[0]
    event_df = play_ex.loc[index-5:index+5,:]
    event_index = event_df['s'].idxmax()
    
    if (event_index == event_df.index[-1]) or (event_index == event_df.index[-2]):
        event_df = play_ex.loc[index-10:index+10,:]
        
    #frame_id = play_ex.loc[event_index]['frameId']
    
    return event_df, event_index