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