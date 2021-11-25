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

def make_field_goal(play_df, players_df):
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