# Understanding Special Teams Plays through Topological Data Analysis (TDA)

Special Teams Play has the ability to significantly impact the outcome of a game in the National Football League (NFL), yet quantitative analysis of special teams has been relatively limited. The current trend of using advanced metrics and data analytics in American football can help stakeholders, such as NFL analysts and coaches, better understand what features significantly influence special teams play. This is the goal of the 2022 NFL Big Data Bowl Challenge. Our team worked with a research scientist from MinedXAI, an exploratory artificial intelligence company, to quantify special teams play for this competition.

This project may also be of interest to anyone who enjoys sports analytics or fantasy football.

Our project is divided into two phases:

Phase 1 of this project is for Elizabeth G. Campolongo, Ranthony A.C. Edmonds, and Chaya Norton's Erdős Data Science Project in Fall 2021, in collaboration with Kenneth Howe (from Mined XAI).

Phase 2 of this project will be completed as a submission to the NFL Big Data Bowl 2022, listed on Kaggle.com: https://www.kaggle.com/c/nfl-big-data-bowl-2022/overview.

Additionally, this project was suggested by MinedXAI, with the goal to leverage Topological Data Analysis (TDA) as a tool to better understand NFL Special Teams Plays. 

Our NFL data was compiled by the hosts of the competition and is described below.
<!-- and the Weather data is from Thomas Bliss (ThompsonJamesBliss on GitHub), one of the competition hosts. -->

# Phase 1: Understanding Field Goals and Extra Points

In Phase 1 of the project, we focused on analyzing field goal and extra point special teams plays. Unlike punt returns and kickoffs, there is a low amount of variation amongst the plays, techniques, and outcomes. 

## Data:

Note we have 7 files from the competetition hosts:
  - Game Data: games.csv [size].
    - The identifying data for each game (season, date & time, location, home and visitor teams).
    - Key variable is `gameId`.
  - PFF Scouting Data: PFFScoutingData.csv [size].
    - Information about kick types, directions, and air time throughout the game.
    - Key variables are `gameId` and `playId`. Note: `nflId` not included.
    - Not used in Phase 1, this is all about punts and kickoffs (kick-types and strategies) and is heavily categorical.
  - Player Data: players.csv [2732 rows × 7 columns].
    - Information for each player (height, weight, birth, college, position, name).
    - Key variable is `nflId`.
  - Play Data: plays.csv [19979 rows × 25 columns].
    - Game-specific temporal information, type of play and play result
    - Key variables are `gameId` and `playId`. Note:`kickerId` is `nflId` of kicker.
  - Tracking Data: tracking2018.csv [12777351 rows x 18 columns], tracking 2019.csv [12170933 rows x 18 columns], and tracking2020.csv [11821701 rows x 18 columns]. 
    - Each contains location and directional data for every player on the field as well as the location and movement of the football for all special teams plays per game from the indicated season.
    - Key variables are `gameID`, `nflId`, and `playId`.

## Preprocessing:

  - Tracking: We subdivide each of the tracking datasets into play type specific files to reduce the memory consumption of the large data files. We then isolate the tracking information relating only to the football, dropping columns with information not available for the football itself (eg., direction and angle), and recombining into a single dataframe by play-type encompassing all three years of data.
  - Play: For the play dataset, we set the gameclock to overall game time measured in seconds, and fill null values in `penaltyYards` and `penaltyCodes` with "0" and "no penalty", respectively.
  - Tracking (take 2): Additionally, we must clean the data to remove instances low-quality tracking data (eg., by identifying the football defying the laws of physics). This is specifically run AFTER the football tracking datasets have been created and the play data has been preprocessed.
  - Player: The player data requires preprocessing to standardize the height measurements. 

## Constructing Play Type DataFrames:
  
In order to perform our analysis on the specific types of plays, we must assemble dataframes of the relevant information for each play type (field goals and extra points). For clustering, we further reduce the size of the extra point and field goal dataframes to include only those features we wish to be considered in the clustering. Maintaining a separate dataframe with all relevant data to which we may append the cluster Id's as a column afterwards allows us to further explore the data points in the clusters.
  - Extra Point: 
    - Include only `specialTeamsPlayType` "Extra Point", so this column is also removed.
    - Merge with player data for kicker ('height', 'weight', 'position', 'displayName') on `nflId` and `kickerId`.
    - Key variables are `gameId`, `playId`, `kickerId` (the kicker's `nflId`).
    - Final columns for clustering are:
      - `yardlineNumber`: Line-of-scrimmage yard line (numeric)
      - `gameClockSeconds`: Overall game time measured in seconds (numeric)
      - `penaltyCodes`: NFL penalty categorization (https://operations.nfl.com/the-rules/2021-nfl-rulebook/#table-of-foul-codes) of penalty on play. Standard penalty code with a "d" at the end indicates this is a penalty on the defense (categorical). 
      - `penaltyYards`: Number of yards gained by the `possessionTeam` through the penalty (numeric)
      - `preSnapHomeScore`: Home team's score prior to the play (numeric)
      - `preSnapVisitorScore`: Visting team's score prior to the play (numeric)
      - `kicker_height`: Height of kicker in inches (numeric)
      - `kicker_weight`: Weight of kicker in pounds (numeric)  
  <!-- Extra informational columns not included in clustering:
      - `playDescription`: Text description of the play
      - `quarter`: Quarter of the game in which the play occurs (numeric)
      - `possessionTeam`: Team kicking the ball
      - `specialTeamsResult`: "Kick Attempt Good", "Kick Attempt No Good", "Blocked Kick Attempt", "Non-Special Teams Result"
      - `kickBlockerId`: `nflId` of blocker
      - `yardlineSide`:
      - `penaltyJerseyNumbers`: Jersey number and team code of player who committed the penalty (multiple players separated by ";")
      - `passResult`: Scrimmage outcome of "Non-Special Teams Result": ("C": Complete pass, "I": Incomplete Pass, "S": Quarterback sack, "IN": Intercepted pass, "R": Scramble, "' '": Designed Rush)
      - `absoluteYardlineNumber`: Location of ball downfield in tracking data coordinates (numeric)
      - `kicker_position`: Position of kicker
      - `kicker_name`: Name of kicker -->
  - Field Goal: 
    - Include only `specialTeamsPlayType` "Field Goal", so this column is also removed.
    - Merge with player data for kicker ('height', 'weight', 'position', 'displayName') on `nflId` and `kickerId`.
    - Key variables are `gameId`, `playId`, `kickerId` (the kicker's `nflId`).
    - Final columns for clustering are:
      - `down`: Down on which the play occurs (numeric)
      - `yardsToGo`: Distance required for first down (numeric)
      - `yardlineNumber`: Line-of-scrimmage yard line (numeric)
      - `gameClockSeconds`: Overall game time measured in seconds (numeric)
      - `penaltyCodes`: NFL penalty categorization (https://operations.nfl.com/the-rules/2021-nfl-rulebook/#table-of-foul-codes) of penalty on play. Standard penalty code with a "d" at the end indicates this is a penalty on the defense (categorical).
      - `penaltyYards`: Number of yards gained by the `possessionTeam` through the penalty (numeric)
      - `preSnapHomeScore`: Home team's score prior to the play (numeric)
      - `preSnapVisitorScore`: Visting team's score prior to the play (numeric)
      - `kickLength`: Distance the ball travels in the air (numeric)
      - `playResult`: Net yards gained by kicking team, penalty yardage inclduded (numeric)
      - `kicker_height`: Height of kicker in inches (numeric)
      - `kicker_weight`: Weight of kicker in pounds (numeric) 
<!-- Extra informational columns not included in clustering:     
      - `playDescription`: Text description of the play
      - `quarter`: Quarter of the game in which the play occurs (numeric)
      - `possessionTeam`: Team kicking the ball
      - `specialTeamsResult`: "Kick Attempt Good", "Kick Attempt No Good", "Blocked Kick Attempt", "Non-Special Teams Result"
      - `kickBlockerId`: `nflId` of blocker
      - `yardlineSide`:
      - `penaltyJerseyNumbers`: Jersey number and team code of player who committed the penalty (multiple players separated by ";")
      - `passResult`: Scrimmage outcome of "Non-Special Teams Result": ("C": Complete pass, "I": Incomplete Pass, "S": Quarterback sack, "IN": Intercepted pass, "R": Scramble, "' '": Designed Rush)
      - `absoluteYardlineNumber`: Location of ball downfield in tracking data coordinates (numeric)
      - `kicker_position`: Position of kicker
      - `kicker_name`: Name of kicker -->

## Feature Engineering:

  - Endzone y-position: y-coordinate of the ball as it crosses the endzone.
  - Expected Endzone y-position: Expected y-coordinate of the ball as it crosses the endzone. This is calculated as a straight line from the position when kicked and the position two frames later (.2 seconds later).
    - Used to calculate the endzone y-error.
  - Endzone y Error: Difference between the expected endzone y-postition and the actual enzone y-position.
  - Off-Center: The distance from the center of the field goal (along the y-axis) of the football as it crosses the endzone.
  - Kicker Core-Distance: How far the kicker has to look to see the closest k number of players at the time of the kick. For instance, in the image below, kicker core-distance k=3 would be the distance from the football to the player in the middle.
    - Note: There may be multiple kicker_core_dist columns depending on how many core-distances are calculated.
<img width="330" alt="KickerCoreDist" src="https://user-images.githubusercontent.com/38985481/144290758-7614f600-deb5-40c2-bfd8-f4cafccee145.png">


## Clustering:

### Step 1: UMAP

UMAP (Uniform Manifold Approximation and Projection) is a topologically derived dimensionality reduction algorithm.

While it is nice that we can reduce the number of dimensions in our data, that is not the primary benefit we glean from it. For our purposes, UMAP is beneficial for two reasons:
1. UMAP is able to cleanly separate regions of local connectedness in data while capturing global structure as well as possible.
2. UMAP allows us to synthesize different notions of distance that apply separately to numeric and categorical data types, helping alleviate some of the pitfalls common to using distance-metric-based clustering algorithms on mixed-type data.

### Step 2: HDBSCAN


  <!--For clustering, we further reduce the size of the extra point and field goal dataframes to include only those features we wish to be considered in the clustering. Maintaining a separate dataframe with all relevant data to which we may append the cluster Id's as a column afterwards allows us to further explore the data points in the clusters.
  - Extra point columns for clustering are:
    - `endzone_y`: The y-position of the football as it crosses the endzone (numeric)
    - `endzone_y_error`: The difference between the actual and expected y-position of the football as it crosses the endzone (numeric)
    - `endzone_y_off_center`: How far the football is from the center of the field goal post as it crosses the endzone (numeric)
    - `kicker_core_dist_{k}`: Core-distance "k" at time of kick.
      - Note: There may be multiple kicker_core_dist columns depending on how many core-distances are calculated.
  - Field goal columns for clustering are:
    - `down`: Down on which the play occurs (numeric)
    - `yardsToGo`: Distance required for first down (numeric)
    - `yardlineNumber`: Line-of-scrimmage yard line (numeric)
    - `gameClockSeconds`: Overall game time measured in seconds (numeric)
    - `penaltyCodes`: NFL penalty categorization (https://operations.nfl.com/the-rules/2021-nfl-rulebook/#table-of-foul-codes) of penalty on play. Standard penalty code with a "d" at the end indicates this is a penalty on the defense (categorical).
    - `penaltyYards`: Number of yards gained by the `possessionTeam` through the penalty (numeric)
    - `preSnapHomeScore`: Home team's score prior to the play (numeric)
    - `preSnapVisitorScore`: Visting team's score prior to the play (numeric)
    - `kickLength`: Distance the ball travels in the air (numeric)
    - `playResult`: Net yards gained by kicking team, penalty yardage inclduded (numeric)
    - `kicker_height`: Height of kicker in inches (numeric)
    - `kicker_weight`: Weight of kicker in pounds (numeric) 
    - `endzone_y`: The y-position of the football as it crosses the endzone (numeric)
    - `endzone_y_error`: The difference between the actual and expected y-position of the football as it crosses the endzone (numeric)
    - `endzone_y_off_center`: How far the football is from the center of the field goal post as it crosses the endzone (numeric)
    - `kicker_core_dist_{k}`: Core-distance "k" at time of kick.
      - Note: There may be multiple kicker_core_dist columns depending on how many core-distances are calculated.
-->

# Phase 2: Understanding Punts and Kickoffs

Coming soon to a GitHub near you! (this one)
