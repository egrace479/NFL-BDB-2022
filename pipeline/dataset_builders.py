import pandas as pd

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

def make_field_goal(play_df, players_df, fg_tracking_ball):
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
    # Remove extraneous columns
    fg = play_fieldgoal.drop(columns =['kickReturnYardage', 'specialTeamsPlayType'])
   
    # Add in kickers
    fg_play = pd.merge(fg, players_df[['nflId', 'height', 'weight','Position', 'displayName']], how = 'left',
             left_on = 'kickerId', right_on = 'nflId')
    fg_plays=fg_play.rename(columns = {"height": 'kicker_height', "weight": 'kicker_weight', "Position": 'kicker_position', "displayName": 'kicker_name'})
    fg_plays=fg_plays.drop(columns=['nflId'])

    # Limit to results with relevant event data
    fg_plays = fg_plays[fg_plays['specialTeamsResult'].isin(['Kick Attempt Good', 'Kick Attempt No Good'])]
    attempts_ids = fg_plays[['gameId', 'playId']]
    attempts_tracking = pd.merge(attempts_ids, fg_tracking_ball, left_on = ['gameId', 'playId'], right_on = ['gameId', 'playId'])
    attempts_event = attempts_tracking[attempts_tracking['event']=='field_goal_attempt']
    attempts_event['mergeId'] = attempts_event['gameId'].astype(str) + attempts_event['playId'].astype(str)
    attempts_ids['mergeId'] = attempts_ids['gameId'].astype(str) + attempts_ids['playId'].astype(str)
    indices_to_drop = attempts_ids[~attempts_ids['mergeId'].isin(attempts_event['mergeId'])].index
    fg_plays = fg_plays.drop(indices_to_drop)

    return fg_plays

def make_extra_point(play_df, players_df, ep_tracking_ball):
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

    # Limit to results with relevant event data
    ep_plays = ep_plays[ep_plays['specialTeamsResult'].isin(['Kick Attempt Good', 'Kick Attempt No Good'])]
    attempts_ids = ep_plays[['gameId', 'playId']]
    attempts_tracking = pd.merge(attempts_ids, ep_tracking_ball, left_on = ['gameId', 'playId'], right_on = ['gameId', 'playId'])
    attempts_event = attempts_tracking[attempts_tracking['event']=='extra_point_attempt']
    attempts_event['mergeId'] = attempts_event['gameId'].astype(str) + attempts_event['playId'].astype(str)
    attempts_ids['mergeId'] = attempts_ids['gameId'].astype(str) + attempts_ids['playId'].astype(str)
    indices_to_drop = attempts_ids[~attempts_ids['mergeId'].isin(attempts_event['mergeId'])].index
    ep_plays = ep_plays.drop(indices_to_drop)

    return ep_plays