# NFL-BDB-2022

This repository is for Elizabeth G. Campolongo, Ranthony A.C. Edmonds, and Chaya Norton's Erdös Data Science Project in Fall 2021, in collaboration with Kenneth Howe (from Mined XAI). 

This project will be completed as a submission to the NFL Big Data Bowl 2022, listed on Kaggle.com: https://www.kaggle.com/c/nfl-big-data-bowl-2022/overview.

Additionally, this project was suggested by MinedXAI, with the goal to leverage Topological Data Analysis (TDA) as a tool to better understand NFL Special Teams Plays. 

Our NFL data was compiled by the hosts of the competition, and the Weather data is from Thomas Bliss (ThompsonJamesBliss on GitHub), one of the competition hosts. 

## Data:

Note we have 7 files from the competetition hosts:
  - Game Data: games.csv, [size]
    - The identifying data for each game (season, date & time, location, home and visitor teams).
    - Key variable is `gameId`.
  - PFF Scouting Data: PFFScoutingData.csv, [size]
    - Information about kick types, directions, and air time throughout the game.
    - Key variables are `gameId` and `playId`. Note: `nflId` not included.
  - Player Data: players.csv, [size]
    - Information for each player (height, weight, birth, college, position, name).
    - Key variable is `nflId`.
  - Play Data: plays.csv, [size]
    - Game-specific temporal information, type of play and play result
    - Key variables are `gameId` and `playId`. Note:`kickerId` is `nflId` of kicker.
  - Tracking Data: tracking2018.csv, tracking 2019.csv, and tracking2020.csv. [size]
    - Each contains location and directional data for every player on the field as well as the location and movement of the football for all special teams plays per game from the indicated season.
    - Key variables are `gameID`, `nflId`, and `playId`.

## Preprocessing:

  - Tracking: We subdivide each of the tracking datasets into play type specific files to reduce the memory consumption of the large data files. We then isolate the tracking information relating only to the football, dropping columns with information not available for the football itself (eg., direction and angle), and recombining into a single dataframe by play-type encompassing all three years of data.
  - Play: For the play dataset, we set the gameclock to overall game time measured in seconds, and fill null values in `penaltyYards` and `penaltyCodes` with "0" and "no penalty", respectively.
  - Tracking (take 2): Additionally, we must clean the data to remove instances low-quality tracking data (eg., by identifying the football defying the laws of physics). This is specifically run AFTER the football tracking datasets have been created and the play data has been preprocessed.
  - Player: The player data requires preprocessing to standardize the height measurements. 

## Constructing Play Type DataFrames:
  
In order to perform our analysis on the specific types of plays, we must assemble dataframes of the relevant information for each play type (field goals and extra points). 
  - Extra Point: 
    - Include only `specialTeamsPlayType` "Extra Point", so this column is also removed.
    - Merge with player data for kicker ('height', 'weight','Position', 'displayName') on `nflId` and `kickerId`.
    - Key variables are `gameId`, `playId`, `kickerId` (the kicker's `nflId`).
    - Final columns are: 
      - `playDescription`: Text description of the play
      - `quarter`: Quarter of the game in which the play occurs (numeric)
      - `possessionTeam`: Team kicking the ball
      - `specialTeamsResult`: "Kick Attempt Good", "Kick Attempt No Good", "Blocked Kick Attempt", "Non-Special Teams Result"
      - `kickBlockerId`: `nflId` of blocker
      - `yardlineSide`:
      - `yardlineNumber`: Line-of-scrimmage yard line (numeric)
      - `gameClockSeconds`: Overall game time measured in seconds (numeric)
      - `penaltyCodes`: NFL penalty categorization (https://operations.nfl.com/the-rules/2021-nfl-rulebook/#table-of-foul-codes) of penalty on play. Standard penalty code with a "d" at the end indicates this is a penalty on the defense. 
      - `penaltyJerseyNumbers`: Jersey number and team code of player who committed the penalty (multiple players separated by ";")
      - `penaltyYards`: Number of yards gained by the `possessionTeam` through the penalty (numeric)
      - `preSnapHomeScore`: Home team's score prior to the play (numeric)
      - `preSnapVisitorScore`: Visting team's score prior to the play (numeric)
      - `passResult`: Scrimmage outcome of "Non-Special Teams Result": ("C": Complete pass, "I": Incomplete Pass, "S": Quarterback sack, "IN": Intercepted pass, "R": Scramble, "' '": Designed Rush)
      - `absoluteYardlineNumber`: Location of ball downfield in tracking data coordinates (numeric)
      - `kicker_height`: Height of kicker in inches (numeric)
      - `kicker_weight`: Weight of kicker in pounds (numeric)
      - `kicker_position`: Position of kicker
      - `kicker_name`: Name of kicker
  - Field Goal: 
    - Include only `specialTeamsPlayType` "Field Goal", so this column is also removed.
    - Merge with player data for kicker ('height', 'weight','Position', 'displayName') on `nflId` and `kickerId`.
    - Key variables are `gameId`, `playId`, `kickerId` (the kicker's `nflId`).
    - Final columns are: 
      - `playDescription`: Text description of the play
      - `quarter`: Quarter of the game in which the play occurs (numeric)
      - `down`: Down on which the play occurs (numeric)
      - `yardsToGo`: Distance required for first down (numeric)
      - `possessionTeam`: Team kicking the ball
      - `specialTeamsResult`: "Kick Attempt Good", "Kick Attempt No Good", "Blocked Kick Attempt", "Non-Special Teams Result"
      - `kickBlockerId`: `nflId` of blocker
      - `yardlineSide`:
      - `yardlineNumber`: Line-of-scrimmage yard line (numeric)
      - `gameClockSeconds`: Overall game time measured in seconds (numeric)
      - `penaltyCodes`: NFL penalty categorization (https://operations.nfl.com/the-rules/2021-nfl-rulebook/#table-of-foul-codes) of penalty on play. Standard penalty code with a "d" at the end indicates this is a penalty on the defense. 
      - `penaltyJerseyNumbers`: Jersey number and team code of player who committed the penalty (multiple players separated by ";")
      - `penaltyYards`: Number of yards gained by the `possessionTeam` through the penalty (numeric)
      - `preSnapHomeScore`: Home team's score prior to the play (numeric)
      - `preSnapVisitorScore`: Visting team's score prior to the play (numeric)
      - `passResult`: Scrimmage outcome of "Non-Special Teams Result": ("C": Complete pass, "I": Incomplete Pass, "S": Quarterback sack, "IN": Intercepted pass, "R": Scramble, "' '": Designed Rush)
      - `kickLength`: Distance the ball travels in the air (numeric)
      - `playResult`: Net yards gained by kicking team, penalty yardage inclduded (numeric)
      - `absoluteYardlineNumber`: Location of ball downfield in tracking data coordinates (numeric)
      - `kicker_height`: Height of kicker in inches (numeric)
      - `kicker_weight`: Weight of kicker in pounds (numeric)
      - `kicker_position`: Position of kicker
      - `kicker_name`: Name of kicker

## Feature Engineering:

  - Endzone y-position: y-coordinate of the ball as it crosses the endzone
  - Expected Endzone y-position: expected y-coordinate of the ball as it crosses the endzone (calculated as a straight line from the position when kicked and the position two frames later (.2 seconds later))
  - Off-Center: the distance from the center of the field goal (along the y-axis) of the football as it crosses the endzone
  - Kicker Core-Distance: 


