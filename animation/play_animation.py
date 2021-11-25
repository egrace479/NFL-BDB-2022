import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from IPython.display import HTML

from querying.tracking_query import get_play, get_play_frame

def draw_pitch(game_id=None, games=None):
    '''
    Draws a basic football pitch using matplotlib.

    Parameters:
    -----------
    game_id - ID of game to draw pitch for
    games - DataFrame containing data from NFL BigDataBowl games.csv

    Returns:
    --------
    fig - The matplotlib figure containing the drawn pitch
    ax - The axes of the matplotlib object for future drawing
    '''
    
    fig, ax = plt.subplots(1,1)

    fig.set_figheight(9)
    fig.set_figwidth(20)

    ax.axis('off')

    if game_id and games is not None:
        home_abbr = games[games['gameId'] == game_id]['homeTeamAbbr'].values[0]
        away_abbr = games[games['gameId'] == game_id]['visitorTeamAbbr'].values[0]
    else:
        home_abbr = 'HOME'
        away_abbr = 'AWAY'

    # Home end zone
    ax.add_patch(Rectangle((0,0), 10, 53, color='#BB0000', ec='w', lw=2, zorder=0))
    ax.text(5, 53/2, home_abbr, color='w', fontsize=40, rotation=90, ha='center', va='center', zorder=0)

    ax.add_patch(Rectangle((110,0), 10, 53, color='#00274C', ec='w', lw=2, zorder=0))
    ax.text(115, 53/2, away_abbr, color='w', fontsize=40, rotation=270, ha='center', va='center', zorder=0)

    ax.add_patch(Rectangle((10,0), 100, 53, color='#67A159', ec='w', lw=2, zorder=0))

    # Add 1-yard lines
    for i in range(10, 110):
        ax.axvline(x=i, ymin=0.05, ymax=0.08, color='w', zorder=0)
        ax.axvline(x=i, ymin=0.33, ymax=0.36, color='w', zorder=0)
        ax.axvline(x=i, ymin=0.64, ymax=0.67, color='w', zorder=0)
        ax.axvline(x=i, ymin=0.92, ymax=0.95, color='w', zorder=0)

    # Add 5-yard lines
    for i in range(15, 110, 5):
        ax.axvline(x=i, color='w', zorder=0)

    # Add 10-yard markers
    for i in range(20, 110, 10):
        ax.text(i, 53-10, str(50-np.abs(i-60)), color='w', size=22, ha='center', va='center', zorder=0)
        ax.text(i-.4, 10, str(50-np.abs(i-60)), color='w', size=22, rotation=180, ha='center', va='center', zorder=0)
    
    # Add goal posts
    y_top = 53 / 2 + (18.5 / 3)
    y_bottom = 53 / 2 - (18.5 / 3)
    ax.axhline(y=y_top, xmin=.045, xmax=0.05, color='w', zorder=0) # Home
    ax.axhline(y=y_bottom, xmin=.045, xmax=0.05, color='w', zorder=0)
    ax.axhline(y=y_top, xmin=0.95, xmax=.955, color='w', zorder=0) # Away
    ax.axhline(y=y_bottom, xmin=0.95, xmax=.955, color='w', zorder=0)

    # Add lines of scrimmage
    ax.axvline(x=12, ymin=.47, ymax=.53, color='w', zorder=0)
    ax.axvline(x=108, ymin=.47, ymax=.53, color='w', zorder=0)
    
    return fig, ax

def animate_play(game_id, play_id, tracking, games=None, save_to=None, as_html=False):
    '''
    Animates a single play from the NFL BigDataBowl dataset.

    Parameters:
    -----------
    game_id - ID of game to draw pitch for
    play_id - ID of play to animate
    tracking - DataFrame containing data from NFL BigDataBowl tracking files
    games - DataFrame containing data from NFL BigDataBowl games.csv
    save_to - Filepath to save animation to
    as_html- Return animation as HTML video

    Returns:
    --------
    fig - The matplotlib figure containing the drawn pitch
    ax - The axes of the matplotlib object for future drawing
    '''

    # Initialize figure
    fig, ax = draw_pitch(game_id, games)

    home_plot, = ax.plot([], [], 'o', markerfacecolor='red', markeredgecolor='w', markersize=15)
    away_plot, = ax.plot([], [], 'o', markerfacecolor='blue', markeredgecolor='w', markersize=15)
    ball_plot, = ax.plot([], [], 'o', markerfacecolor='black', markeredgecolor='w', markersize=10)

    drawings = [home_plot, away_plot, ball_plot]

    # Get play
    play = get_play(game_id, play_id, tracking)

    # Get play frames
    frames = play['frameId'].unique()

    # Define init animation function
    def init():
        home_plot.set_data([], [])
        away_plot.set_data([], [])
        ball_plot.set_data([], [])

        return drawings

    # Define primary animation function
    def animate(i):
        home, away, ball = get_play_frame(play, i)
        
        home_plot.set_data(home['x'], home['y'])
        away_plot.set_data(away['x'], away['y'])
        ball_plot.set_data(ball['x'], ball['y'])

        return drawings
   
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)

    # Optional arguments
    if as_html:
        return HTML(anim.to_html5_video())
    if save_to:
        video_writer = FFMpegWriter(fps=30)
        anim.save(save_to, video_writer)

    return anim