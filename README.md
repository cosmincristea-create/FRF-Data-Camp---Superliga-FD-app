# FRF-Data-Camp---Superliga-FD-app
This app is part of our project proposal for the FRF Data Camp 2025. We (Felix and I) wanted to look at developing a solution for coaches, analysts and executives working within the professional football clubs in Romania and give them a tool that can bring the complex Wyscout data to life. **What you see here is the fruition of that idea.**

This app allows you to **visualise** different types of datasets, from player advanced stats to team advanced stats, physical data and event data. 
**Tactical insights** can be gathered from using this app as you can now look to create shot maps, pass maps and networks, heatmaps and defensive actions analysis. 

For **scouting** needs the app has an in-built pizza plot maker where you can either use the default template or make your own analysis from a 40+ list of recognised Wyscout metrics. You can go a step further and compare players using scatter plots by filtering through their position, club and a wide range of metrics to build an in-depth report for your scouting needs.

**Please note that this app was built to handle Wyscout data and the resulting csv files**. 
Usage of other data providers will cause the app to crash.

Superliga Analytics Dashboard project documentation 

1. Project overview & technical architecture
The Superliga Analytics Dashboard is an interactive web application developed in Python, designed for the analysis of football data. The application is built on the Streamlit framework, which serves as the frontend for user interaction and visualisation rendering. 

The core logic is centered around data processing and visualisation, handled by a suite of football specialised libraries:

For data manipulation: The **Pandas** library is the primary engine for all data manipulation. The application is designed to process user-uploaded CSV files (from Wyscout), which are read into Pandas DataFrames. All filtering, metric calculation, and data restructuring operations are performed within this environment.

Interactive charting: **Plotly Express** is used for generating interactive data visualisations such as scatter plots and bar charts. This allows for dynamic exploration of the data through hovering and tooltips.

Football-specific visualisations: The **mplsoccer** library is used for creating bespoke pitch visualisations, including pass maps, shot maps, heatmaps, and pass networks. This library provides the necessary tools to accurately plot event data onto a Wyscout-standard pitch.

2. Development chronology & Feature logic
The application was developed over three weeks, with functionality added in distinct, logical blocks. The app was built with the core purpose of being useful to a variety of stakeholders from coaching staff to recuritment analysts and higher level executives.

Week 1: Core modules - Statistical analysis
Objective: To establish the core functionality for high-level statistical analysis of seasonal data, processing the DataCamp Wyscout teams and event data Wyscout files.

**Modules built:
**
Team & Player Statistics: The logic for these modules involves reading a statistics CSV into a Pandas DataFrame. A dictionary (metric_mapping) is used to translate raw column names from the source file into user-friendly labels. Visualisations are driven by user selections from** st.selectbox** widgets, which dynamically filter the DataFrame to generate rankings or create scatter plots comparing two selected metrics.

Week 2: Advanced visualisations & Event Data processing
Objective: To give the app the tools to build summary statistics and begin processing granular, single-match event data.

Modules built:

**Pizza Plots**: This feature uses the **mplsoccer.PyPizza** class. The core logic involves calculating a player's percentile rank for a given set of KPIs against their positional peers. This is achieved using the **scipy.stats.percentileofscore** function on the relevant DataFrame columns. The results are then passed to the PyPizza class for rendering and visualisation.

**Event Data Dashboard** (v1): This marked the introduction of event data processing. The logic is based on filtering the main event DataFrame by the **type.primary column** (e.g., 'shot', 'pass') and then plotting the corresponding **location.x** and **location.y** coordinates onto a pitch object created with **mplsoccer.Pitch.**

Week 3: Feature upgrades & advanced logic (and debugging)
Objective: To enhance existing tools with more advanced features and address data-specific challenges, as well as debug any issues within the code.

Key features & fixes:

**Interactive Pass Networks & Multi-Player Maps**: The create_pass_network function was updated to accept a **min_pass_count parameter**, allowing the DataFrame of pass combinations to be filtered before plotting. The create_player_pass_maps_grid function was introduced, utilising **mplsoccer.Pitch.grid()** to dynamically create a figure with multiple subplots based on the number of selected players.

Player Actions Analysis (beta): This feature aims to combine two visualisation techniques on a single mplsoccer axis. First, a heatmap is generated using pitch.heatmap() on a 2D histogram of all action locations, smoothed with scipy.ndimage.gaussian_filter. Second, individual actions are overlaid as distinct markers using pitch.scatter(). This feature is still in beta phase and is currently being tested in an offline environment. When ready, this will be deployed onto the live app here.

Bug fix - Inferring Duel Wins: A key challenge was the absence of an explicit "duel won" flag in the event data, as Wyscout does not track the success of a duel like Opta or StatsBomb. This meant that the app had to get clever when working with the duel data from the CSV files. The solution was to infer the outcome based on the event sequence. The entire match DataFrame is sorted by timestamp (minute, second), and a new column is created using pandas.DataFrame.shift(-1) to get the team associated with the subsequent event. A duel is then flagged as "won" if the team that initiated the duel is the same as the team of the next event, indicating a successful possession regain.

3. Limitations & technical considerations
Data Schema Dependency: The application's data processing logic is tightly coupled to the Wyscout data schema. Column names like type.primary, location.x, and pass.accurate are hard-coded. An upload from a different data provider (e.g., Opta, StatsBomb) would cause a KeyError and crash the app. A future improvement would be to implement a data mapping interface to abstract these column names.

Scalability: The current implementation uses Pandas to load the entire user-uploaded CSV file into memory. This approach is efficient for single-match or single-season files but is not scalable to larger, multi-season datasets, which could lead to memory overflow issues on standard hosting environments. Future optimisation should explore more memory-efficient data processing backends like Polars or DuckDB.

Analysis scope: All analysis is currently static and post-match. The application is not designed for live data ingestion or real-time analysis.

4. Future Roadmap & potential upgrades
The current application serves as a strong foundation for several advanced features. We believe this app is useful in a plug-and-play scenario for different stakeholders.

Time-series & Trend analysis: Implement functionality to track key metrics over time. This would involve grouping data by match date and plotting metrics over a rolling window (e.g., a 5-game rolling average of a team's xG) to visualise performance trends. This would allow coaching staff members to asses team performance during the season and have data embedded in their training periodisation workflows.

Data mix (Physical & Eventdata): A significant upgrade would be to merge the physical data (sprints, distance covered) with the event data on a player-by-player and match-by-match basis. This would enable deeper performance analysis, such as correlating a player's physical output with their technical execution (e.g., passing accuracy vs. distance covered). Very useful for both recuritment purposes, when assessing a potential target, and when doing performance analysis on your own team.

Automated PDF reporting: Develop a module that uses a library like FPDF or ReportLab to generate "one-click" PDF reports. This would involve programmatically creating the visualisations for a selected team or player and compiling them into a structured, shareable document.

Tactical & formation analysis: By parsing team formation data for different periods of a match, we could filter event data accordingly. This would allow for comparative analysis of tactical setups, such as visualising how a team's pass network or defensive shape changes after switching from a 4-4-2 to a 3-5-2. This one sits at the end because of the limitations of the data fed to the app. However, this could be integrated into an ML project that focuses on optical tracking to get the data from video. The optical tracking solution should be able to distinguish formations based on instructions and then send that information into a csv file which would then be fed into the app to perform the formation analysis.
