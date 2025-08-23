import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, PyPizza, VerticalPitch
from scipy.stats import percentileofscore
from scipy.ndimage import gaussian_filter
import plotly.express as px
from datetime import datetime
import io
import base64
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Superliga Analytics Dashboard",
    page_icon="🇷🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING AND HELPER FUNCTIONS ---

# Custom CSS with IBM Color Palette
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #648FFF; /* IBM Blue */
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #DC267F; /* IBM Magenta */
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def create_pizza_plot(player_values, params, player_info, slice_colors, param_colors, show_legend=True):
    """Create a color-coded pizza plot, with an option to show/hide the category legend."""
    baker = PyPizza(
        params=params,
        background_color="#222222",
        straight_line_color="#EBEBE9", straight_line_lw=1,
        last_circle_lw=1, last_circle_color="#EBEBE9",
        other_circle_ls="-.", other_circle_lw=1,
        inner_circle_size=20
    )

    kwargs_values_to_use = dict(
        color="#EBEBE9", fontsize=12, zorder=3,
        bbox=dict(edgecolor="#EBEBE9", facecolor="#252525", boxstyle="round,pad=0.2", lw=1)
    )

    fig, ax = baker.make_pizza(
        player_values,
        figsize=(8, 10),
        param_location=110,
        kwargs_slices=dict(facecolor=slice_colors, edgecolor="#EBEBE9", zorder=2, linewidth=1),
        kwargs_params=dict(color=param_colors, fontsize=12, va="center"),
        kwargs_values=kwargs_values_to_use
    )

    # Display team name and player info in title
    fig.text(0.515, 0.97, f"{player_info['name']} | Age: {player_info['age']}", size=20,
             ha="center", color="#EBEBE9", fontweight='bold')
    fig.text(0.515, 0.94, f"{player_info['team']} | {player_info['position']}", size=15,
             ha="center", color="#EBEBE9")
      
    if show_legend:
        fig.text(0.35, 0.90, "Attacking", size=12, color="#EBEBE9")
        fig.text(0.49, 0.90, "Possession", size=12, color="#EBEBE9")
        fig.text(0.64, 0.90, "Defensive", size=12, color="#EBEBE9")
        fig.patches.extend([
            plt.Rectangle((0.32, 0.90), 0.025, 0.015, fill=True, color="#FE6100", transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.46, 0.90), 0.025, 0.015, fill=True, color="#648FFF", transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.61, 0.90), 0.025, 0.015, fill=True, color="#FFB000", transform=fig.transFigure, figure=fig),
        ])

    fig.text(0.99, 0.06, "FRF Data Camp 2025 project FC", size=8, color="#EBEBE9", ha="right", style='italic')
    fig.text(0.99, 0.04, "inspired by: @Worville and @FootballSlices", size=8, color="#EBEBE9", ha="right", style='italic')
    fig.text(0.99, 0.02, "Data Source: User Upload", size=8, color="#EBEBE9", ha="right", style='italic')
      
    center_circle = plt.Circle((0, 0), 0.12, color='#222222', ec='white', lw=1.5, zorder=4)
    ax.add_patch(center_circle)

    return fig

def create_benchmark_chart(player_name, data):
    """Creates a horizontal bar chart to benchmark a player's percentile ranks."""
    fig, ax = plt.subplots(figsize=(10, len(data) * 0.5))
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#222222')

    bars = ax.barh(data['metric'], data['percentile'], color='#4A4A4A', edgecolor='white')

    for bar in bars:
        if bar.get_width() > 75:
            bar.set_color('#00FF00')
        elif bar.get_width() > 50:
            bar.set_color('#77DD77')
        elif bar.get_width() < 25:
            bar.set_color('#FF6961')

    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.tick_params(axis='y', colors='white', labelsize=12)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')

    for index, row in data.iterrows():
        text_color = 'black' if row['percentile'] > 75 else 'white'
        ax.text(row['percentile'] - 2, index, str(row['percentile']), color=text_color, va='center', ha='right', fontsize=12, fontweight='bold')
        ax.text(0, index, f" {row['metric']} ({row['value']:.2f})", color='white', va='center', ha='left', fontsize=12)

    ax.set_title(f"{player_name} - Percentile Ranks", color='white', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_scatter_plot(data, x_metric, y_metric, title="Scatter Plot", text_column=None, color_column=None):
    fig = px.scatter(
        data, x=x_metric, y=y_metric,
        hover_data=data.columns.tolist(),
        title=title, template='plotly_white',
        color=color_column,
        text=text_column
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(width=800, height=600, title_font_size=16)
    return fig

def create_pass_map(event_data, team_name=None, pass_filter='All Passes', location_filter='All Passes'):
    passes = event_data[event_data['type.primary'] == 'pass'].copy()
    if passes.empty: return None

    if location_filter == 'Forward Passes':
        passes = passes[passes['pass.endLocation.x'] > passes['location.x']]
    elif location_filter == 'Final Third Passes':
        passes = passes[passes['location.x'] > 66.7]
    elif location_filter == 'Own Third Passes':
        passes = passes[passes['location.x'] < 33.3]

    successful_passes = passes[passes['pass.accurate'] == True]
    unsuccessful_passes = passes[passes['pass.accurate'] == False]

    pitch = Pitch(pitch_type='wyscout', pitch_color='#22312b', line_color='#c7d5cc', linewidth=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    ax.set_title(f'{location_filter} for {team_name} ({pass_filter})', fontsize=16, pad=20, color='white')

    if pass_filter == 'All Passes':
        pitch.arrows(successful_passes['location.x'], successful_passes['location.y'], successful_passes['pass.endLocation.x'], successful_passes['pass.endLocation.y'],
                     width=2, headwidth=6, headlength=8, color='#785EF0', ax=ax)
        pitch.arrows(unsuccessful_passes['location.x'], unsuccessful_passes['location.y'], unsuccessful_passes['pass.endLocation.x'], unsuccessful_passes['pass.endLocation.y'],
                     width=2, headwidth=6, headlength=8, color='#FE6100', ax=ax)
    elif pass_filter == 'Successful Only':
        pitch.arrows(successful_passes['location.x'], successful_passes['location.y'], successful_passes['pass.endLocation.x'], successful_passes['pass.endLocation.y'],
                     width=2, headwidth=6, headlength=8, color='#785EF0', ax=ax)
    elif pass_filter == 'Unsuccessful Only':
        pitch.arrows(unsuccessful_passes['location.x'], unsuccessful_passes['location.y'], unsuccessful_passes['pass.endLocation.x'], unsuccessful_passes['pass.endLocation.y'],
                     width=2, headwidth=6, headlength=8, color='#FE6100', ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#785EF0', lw=2, label='Successful Pass'),
        Line2D([0], [0], color='#FE6100', lw=2, label='Unsuccessful Pass')
    ]
    if pass_filter == 'All Passes':
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#22312b', edgecolor='white', labelcolor='white')
    return fig

def create_shot_map(event_data, team_name=None):
    shots = event_data[event_data['type.primary'] == 'shot'].copy()
    if shots.empty:
        st.warning("No shot data found in this match for this team.")
        return None

    pitch = VerticalPitch(half=True, pitch_type='wyscout', pitch_color='#22312b', line_color='#FFFFFF')
    fig, ax = pitch.draw(figsize=(10, 10))
    fig.set_facecolor('#22312b')
    ax.set_title(f'Shot Map for {team_name}', fontsize=16, pad=20, color='white')

    goals = shots[shots['shot.isGoal'] == True]
    on_target = shots[(shots['shot.onTarget'] == True) & (shots['shot.isGoal'] == False)]
    off_target = shots[shots['shot.onTarget'] == False]

    if not off_target.empty:
        pitch.scatter(off_target['location.x'], off_target['location.y'], s=(off_target['shot.xg'] * 500) + 100,
                      color='#B0B0B0', edgecolors='#222222', marker='o',
                      alpha=0.5, ax=ax, label='Off Target')
    if not on_target.empty:
        pitch.scatter(on_target['location.x'], on_target['location.y'], s=(on_target['shot.xg'] * 500) + 100,
                      color='#648FFF', edgecolors='#222222', marker='o',
                      alpha=on_target['shot.postShotXg'].fillna(0.6), ax=ax, label='On Target')
    if not goals.empty:
        pitch.scatter(goals['location.x'], goals['location.y'], s=(goals['shot.xg'] * 500) + 100,
                      color='#DC267F', edgecolors='white', marker='*',
                      alpha=goals['shot.postShotXg'].fillna(0.8), ax=ax, label='Goal')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Off Target', markerfacecolor='#B0B0B0', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='On Target (Opacity=postShotXg)', markerfacecolor='#648FFF', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Goal (Opacity=postShotXg)', markerfacecolor='#DC267F', markersize=15, markeredgecolor='white')
    ]
    ax.legend(handles=legend_elements, loc='upper left', facecolor='#22312b', edgecolor='white', labelcolor='white')
    return fig

def create_pass_network(event_data, team_name=None):
    passes = event_data[(event_data['type.primary'] == 'pass') & (event_data['pass.accurate'] == True) & (event_data['team.name'] == team_name)].copy()
    if passes.empty:
        st.warning("No successful passes found for this team in this match.")
        return None

    avg_locations = passes.groupby('player.name').agg({'location.x': ['mean'], 'location.y': ['mean', 'count']})
    avg_locations.columns = ['x', 'y', 'count']

    pass_combinations = passes.groupby(['player.name', 'pass.recipient.name']).id.count().reset_index()
    pass_combinations.rename({'id': 'pass_count'}, axis='columns', inplace=True)
    pass_combinations = pass_combinations.merge(avg_locations, left_on='player.name', right_index=True)
    pass_combinations = pass_combinations.merge(avg_locations, left_on='pass.recipient.name', right_index=True, suffixes=('', '_end'))

    pitch = Pitch(pitch_type='wyscout', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')
    ax.set_title(f'Pass Network for {team_name}', fontsize=16, pad=20, color='white')

    pitch.lines(pass_combinations.x, pass_combinations.y, pass_combinations.x_end, pass_combinations.y_end,
                lw=pass_combinations.pass_count, color='#FFFFFF', zorder=1, ax=ax, alpha=0.5)
    pitch.scatter(avg_locations.x, avg_locations.y, s=avg_locations['count'] * 10,
                  color='#DC267F', edgecolors='white', linewidth=1, alpha=1, ax=ax)
    for index, row in avg_locations.iterrows():
        pitch.annotate(index, xy=(row.x, row.y), c='white', va='center', ha='center', size=10, ax=ax)

    return fig

def create_heatmap(event_data, team_name=None):
    team_events = event_data[event_data['team.name'] == team_name].copy()
    if team_events.empty:
        st.warning("No events found for this team in this match.")
        return None

    pitch = Pitch(pitch_type='wyscout', pitch_color='#22312b', line_color='#efefef', line_zorder=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')
      
    bin_statistic = pitch.bin_statistic(team_events['location.x'], team_events['location.y'], statistic='count', bins=(6, 5))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
      
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.ax.yaxis.set_tick_params(color='#efefef')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

    ax.set_title(f'Heatmap for {team_name}', fontsize=16, pad=20, color='white')
    return fig

def create_detailed_shot_map(shot_data, team_name):
    """Creates a detailed shot map with accompanying statistical charts."""
    if shot_data.empty:
        st.warning(f"No shot data available for {team_name}.")
        return None, None

    # Main pitch plot for shot locations
    pitch = VerticalPitch(half=True, pitch_type='wyscout', pitch_color='#22312b', line_color='#FFFFFF')
    fig, ax = pitch.draw(figsize=(10, 8))
    fig.set_facecolor('#22312b')
    ax.set_title(f'Detailed Shot Analysis for {team_name}', fontsize=16, pad=20, color='white')

    # Define markers and colors for different body parts
    body_part_map = {
        'Right Foot': {'marker': 'o', 'color': '#648FFF'},
        'Left Foot': {'marker': 'o', 'color': '#FE6100'},
        'Head': {'marker': '^', 'color': '#FFB000'},
        'Other': {'marker': 's', 'color': '#B0B0B0'}
    }

    for index, shot in shot_data.iterrows():
        body_part = shot.get('shot.bodyPart', 'Other')
        marker_style = body_part_map.get(body_part, body_part_map['Other'])
        
        # Use different edgecolors for outcomes
        edge_color = 'white'
        if shot.get('shot.isGoal'):
            edge_color = '#00FF00' # Bright Green for Goal
        elif shot.get('shot.onTarget'):
            edge_color = '#FFFF00' # Yellow for On Target

        pitch.scatter(shot['location.x'], shot['location.y'],
                      s=(shot['shot.xg'] * 500) + 100,
                      ax=ax,
                      marker=marker_style['marker'],
                      facecolor=marker_style['color'],
                      edgecolor=edge_color,
                      lw=2,
                      alpha=0.8)

    # Create a custom legend for body parts and outcomes
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=v['marker'], color='w', label=k, markerfacecolor=v['color'], markersize=10) for k, v in body_part_map.items()]
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', label='Goal', markeredgecolor='#00FF00', markerfacecolor='None', markersize=10, mew=2),
        Line2D([0], [0], marker='o', color='w', label='On Target', markeredgecolor='#FFFF00', markerfacecolor='None', markersize=10, mew=2),
        Line2D([0], [0], marker='o', color='w', label='Off Target', markeredgecolor='white', markerfacecolor='None', markersize=10, mew=2)
    ])
    ax.legend(handles=legend_elements, loc='upper left', facecolor='#22312b', edgecolor='white', labelcolor='white')

    # --- Create secondary charts for more detail ---
    charts = {}
    # Bar chart for shot body part distribution
    if 'shot.bodyPart' in shot_data.columns:
        body_part_counts = shot_data['shot.bodyPart'].value_counts()
        fig_body = px.bar(body_part_counts, x=body_part_counts.index, y=body_part_counts.values,
                          title='Shots by Body Part', labels={'x': 'Body Part', 'y': 'Number of Shots'})
        fig_body.update_layout(template='plotly_dark')
        charts['body_part'] = fig_body

    # Bar chart for goal zone distribution
    if 'shot.goalZone' in shot_data.columns:
        goal_zone_counts = shot_data['shot.goalZone'].value_counts()
        fig_zone = px.bar(goal_zone_counts, x=goal_zone_counts.index, y=goal_zone_counts.values,
                          title='Shot Placements (Goal Zones)', labels={'x': 'Goal Zone', 'y': 'Number of Shots'})
        fig_zone.update_layout(template='plotly_dark')
        charts['goal_zone'] = fig_zone
        
    return fig, charts

def create_possession_flow_map(possession_data, team_name):
    """Visualizes possession sequences on a pitch, colored by outcome."""
    # Get unique possession sequences by their ID
    possession_events = possession_data.dropna(subset=['possession.id']).copy()
    unique_possessions = possession_events.drop_duplicates(subset=['possession.id']).copy()

    if unique_possessions.empty:
        st.warning(f"No possession data available for {team_name}.")
        return None, None

    pitch = Pitch(pitch_type='wyscout', pitch_color='#22312b', line_color='#c7d5cc', linewidth=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')
    ax.set_title(f'Possession Flow & Outcomes for {team_name}', fontsize=16, pad=20, color='white')

    # Categorize possessions by their outcome
    ended_in_goal = unique_possessions[unique_possessions['possession.attack.withGoal'] == True]
    ended_in_shot = unique_possessions[(unique_possessions['possession.attack.withShot'] == True) & (unique_possessions['possession.attack.withGoal'] == False)]
    ended_without_shot = unique_possessions[unique_possessions['possession.attack.withShot'] == False]

    # Plot arrows for each category
    for df, color in zip([ended_in_goal, ended_in_shot, ended_without_shot], ['#00FF00', '#FFFF00', '#FE6100']):
        if not df.empty and 'possession.startLocation.x' in df.columns:
            pitch.arrows(df['possession.startLocation.x'], df['possession.startLocation.y'],
                         df['possession.endLocation.x'], df['possession.endLocation.y'],
                         width=2, headwidth=6, headlength=8, color=color, ax=ax, alpha=0.7)

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00FF00', lw=4, label='Possession leading to a Goal'),
        Line2D([0], [0], color='#FFFF00', lw=4, label='Possession leading to a Shot'),
        Line2D([0], [0], color='#FE6100', lw=4, label='Possession without a Shot')
    ]
    ax.legend(handles=legend_elements, loc='upper left', facecolor='#22312b', edgecolor='white', labelcolor='white')

    # --- NEW: Create a more effective 100% stacked bar chart for flank analysis ---
    charts = {}
    if 'possession.attack.flank' in unique_possessions.columns:
        # 1. Define the outcome for each possession
        def get_outcome(row):
            if row['possession.attack.withGoal']:
                return 'Goal'
            elif row['possession.attack.withShot']:
                return 'Shot (No Goal)'
            else:
                return 'No Shot'
        
        unique_possessions['outcome'] = unique_possessions.apply(get_outcome, axis=1)
        
        # 2. Calculate the percentage of each outcome per flank
        flank_effectiveness = unique_possessions.groupby('possession.attack.flank')['outcome'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
        
        # 3. Create the 100% stacked bar chart
        fig_flank = px.bar(
            flank_effectiveness,
            x="possession.attack.flank",
            y="percentage",
            color="outcome",
            title="Attacking Flank Effectiveness",
            labels={'possession.attack.flank': 'Attacking Flank', 'percentage': 'Percentage of Possessions'},
            text_auto='.1f', # Format text to one decimal place
            color_discrete_map={
                'Goal': '#00FF00',
                'Shot (No Goal)': '#FFFF00',
                'No Shot': '#FE6100'
            }
        )
        fig_flank.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
        fig_flank.update_layout(template='plotly_dark', yaxis_ticksuffix='%')
        charts['flank_analysis'] = fig_flank

    return fig, charts
            

def create_defensive_actions_map(defensive_events, team_name):
    """Creates a pitch map showing the location of successful defensive actions."""
    if defensive_events.empty:
        st.warning("No successful defensive actions found for this team.")
        return None

    pitch = Pitch(pitch_type='wyscout', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')
    ax.set_title(f'Successful Defensive Actions for {team_name}', fontsize=16, pad=20, color='white')

    # Define markers for different action types
    action_markers = {
        'interception': {'marker': 'x', 'color': '#648FFF', 'label': 'Interception'},
        'tackle': {'marker': 's', 'color': '#FE6100', 'label': 'Successful Tackle'},
        'clearance': {'marker': '^', 'color': '#FFB000', 'label': 'Clearance'},
        'block': {'marker': 'p', 'color': '#785EF0', 'label': 'Block'}
    }

    legend_elements = []
    for action, style in action_markers.items():
        action_df = defensive_events[defensive_events['type.primary'] == action]
        if not action_df.empty:
            pitch.scatter(action_df['location.x'], action_df['location.y'],
                          s=100, ax=ax, marker=style['marker'],
                          facecolor=style['color'], edgecolors='white', lw=1, label=style['label'])
            legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], color='w', label=style['label'],
                                              markerfacecolor=style['color'], markersize=10))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', facecolor='#22312b', edgecolor='white', labelcolor='white')
        
    return fig

def create_defensive_heatmap(defensive_events, team_name):
    """Creates a gaussian heatmap of key defensive actions (interceptions & clearances)."""
    key_actions = defensive_events[defensive_events['type.primary'].isin(['interception', 'clearance'])]
    if key_actions.empty:
        st.warning("No interceptions or clearances found for this team.")
        return None

    pitch = Pitch(pitch_type='wyscout', pitch_color='#22312b', line_color='#efefef', line_zorder=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#22312b')
      
    bin_statistic = pitch.bin_statistic(key_actions['location.x'], key_actions['location.y'], statistic='count', bins=(6, 5))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
      
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.ax.yaxis.set_tick_params(color='#efefef')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

    ax.set_title(f'Key Defensive Areas (Interceptions & Clearances) for {team_name}', fontsize=16, pad=20, color='white')
    return fig        

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00FF00', lw=4, label='Possession leading to a Goal'),
        Line2D([0], [0], color='#FFFF00', lw=4, label='Possession leading to a Shot'),
        Line2D([0], [0], color='#FE6100', lw=4, label='Possession without a Shot')
    ]
    ax.legend(handles=legend_elements, loc='upper left', facecolor='#22312b', edgecolor='white', labelcolor='white')

    # --- Create a secondary chart for flank analysis ---
    charts = {}
    if 'possession.attack.flank' in unique_possessions.columns:
        flank_outcomes = unique_possessions.groupby('possession.attack.flank')[['possession.attack.withShot', 'possession.attack.withGoal']].sum().reset_index()
        flank_outcomes.rename(columns={'possession.attack.withShot': 'Ended with Shot', 'possession.attack.withGoal': 'Ended with Goal'}, inplace=True)
        
        fig_flank = px.bar(flank_outcomes, x='possession.attack.flank', y=['Ended with Shot', 'Ended with Goal'],
                           title='Attacking Flank Effectiveness', barmode='group',
                           labels={'possession.attack.flank': 'Attacking Flank', 'value': 'Count'})
        fig_flank.update_layout(template='plotly_dark')
        charts['flank_analysis'] = fig_flank

    return fig, charts

def download_plot(fig, filename):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buffer.seek(0)
    b64 = base64.b64encode(img_buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download Plot</a>'
    return href

def download_plotly(fig, filename):
    img_bytes = fig.to_image(format="png", width=1200, height=800)
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download Plot</a>'
    return href

# --- APP LAYOUT ---
st.markdown('<h1 class="main-header">⚽ Superliga Analytics Dashboard</h1>', unsafe_allow_html=True)
st.sidebar.title("Navigation")
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Team Statistics", "Player Analysis", "Pizza Plots", "Event Data Analysis", "Pass Maps", "Shot Maps", "Pass Networks", "Heatmaps", "Physical Analysis"])

st.sidebar.markdown("### Upload Data Files")
uploaded_files = {}
uploaded_files['team_stats'] = st.sidebar.file_uploader("Upload Team Statistics CSV", type=['csv'], key="team_stats_uploader")
uploaded_files['superliga_players'] = st.sidebar.file_uploader("Upload Superliga Player Stats CSV", type=['csv'], key="superliga_uploader")
uploaded_files['event_data'] = st.sidebar.file_uploader("Upload Event Data CSV", type=['csv'], key="event_data_uploader")
uploaded_files['physical_stats'] = st.sidebar.file_uploader("Upload Physical Metrics CSV", type=['csv'], key="physical_stats_uploader")

# --- MAIN CONTENT AREA ---
if analysis_type == "Team Statistics":
    st.markdown('<h2 class="sub-header">Team Statistics Analysis</h2>', unsafe_allow_html=True)
    if uploaded_files.get('team_stats') is not None:
        team_data = pd.read_csv(uploaded_files['team_stats'])
        st.subheader("Team Dataset Overview")
        st.dataframe(team_data)

        metric_mapping = {
            'total.goals': 'Goals', 'total.assists': 'Assists', 'total.shots': 'Shots', 'total.penalties': 'Penalties',
            'total.linkupPlays': 'Link-up Plays', 'total.duelsWon': 'Duels Won', 'total.fouls': 'Fouls', 'total.passes': 'Passes',
            'total.successfulPasses': 'Successful Passes', 'total.successfulSmartPasses': 'Successful Smart Passes',
            'total.successfulPassesToFinalThird': 'Successful Passes to Final Third', 'total.successfulCrosses': 'Successful Crosses',
            'total.successfulForwardPasses': 'Successful Forward Passes', 'total.successfulBackPasses': 'Successful Back Passes',
            'total.successfulThroughPasses': 'Successful Through Passes', 'total.successfulKeyPasses': 'Successful Key Passes',
            'total.successfulVerticalPasses': 'Successful Vertical Passes', 'total.successfulLongPasses': 'Successful Long Passes',
            'total.successfulDribbles': 'Successful Dribbles', 'total.interceptions': 'Interceptions',
            'total.successfulDefensiveAction': 'Successful Defensive Actions', 'total.recoveries': 'Recoveries',
            'total.xgShot': 'xGShot', 'total.ppda': 'PPDA', 'total.xgShotAgainst': 'xGShot Against',
            'percent.defensiveDuelsWon': 'Defensive Duels Won %', 'percent.offensiveDuelsWon': 'Offensive Duels Won %',
            'percent.successfulPasses': 'Successful Passes %', 'percent.successfulSmartPasses': 'Successful Smart Passes %',
            'percent.shotsOnTarget': 'Shots on Target %', 'percent.successfulDribbles': 'Successful Dribbles %',
            'percent.successfulForwardPasses': 'Successful Forward Passes %', 'percent.successfulThroughPasses': 'Successful Through Passes %',
            'percent.successfulKeyPasses': 'Successful Key Passes %', 'percent.successfulVerticalPasses': 'Successful Vertical Passes %',
            'percent.successfulLongPasses': 'Successful Long Passes %', 'percent.gkAerialDuelsWon': 'GK Aerial Duels Won %',
            'percent.successfulTouchInBox': 'Successful Touch in Box %'
        }
        
        analysis_df = team_data[['team.name']].copy()
        for raw_name, pretty_name in metric_mapping.items():
            if raw_name in team_data.columns:
                analysis_df[pretty_name] = team_data[raw_name]

        if len(analysis_df.columns) < 2:
            st.warning("Warning: No KPIs were found in the uploaded file. Please check that the column names match the expected Wyscout format.")
        
        analysis_view = st.radio("Select Analysis View:", ('Rankings Table', 'Comparison Scatter Plot', 'League Ranking Bar Chart'))

        if analysis_view == 'Rankings Table':
            selectable_metrics = analysis_df.columns[1:].tolist()
            if not selectable_metrics:
                st.error("No metrics are available for analysis. Please check the uploaded CSV file.")
            else:
                selected_metric = st.selectbox("Select metric to rank by:", selectable_metrics)
                st.dataframe(analysis_df[['team.name', selected_metric]].sort_values(by=selected_metric, ascending=False))

        elif analysis_view == 'Comparison Scatter Plot':
            selectable_metrics = analysis_df.columns[1:].tolist()
            if len(selectable_metrics) < 2:
                st.error("At least two metrics must be available for a scatter plot.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    x_metric = st.selectbox("Select X-axis metric", selectable_metrics)
                with col2:
                    y_metric = st.selectbox("Select Y-axis metric", selectable_metrics, index=1)
                fig = create_scatter_plot(analysis_df, x_metric, y_metric, f"{x_metric} vs {y_metric}", text_column='team.name', color_column='team.name')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plotly(fig, f"team_scatter_{x_metric}_vs_{y_metric}"), unsafe_allow_html=True)

        elif analysis_view == 'League Ranking Bar Chart':
            selectable_metrics = analysis_df.columns[1:].tolist()
            if not selectable_metrics:
                st.error("No metrics are available for analysis.")
            else:
                selected_metric = st.selectbox("Select metric to rank by:", selectable_metrics)
                top_10 = analysis_df.nlargest(10, selected_metric)
                fig = px.bar(top_10, x=selected_metric, y='team.name', orientation='h', title=f"Top 10 Teams for {selected_metric}", color='team.name')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plotly(fig, f"team_barchart_{selected_metric}"), unsafe_allow_html=True)
    else:
        st.info("Please upload a team statistics CSV file to begin analysis.")

elif analysis_type == "Player Analysis":
    st.markdown('<h2 class="sub-header">Player Scouting & Analysis</h2>', unsafe_allow_html=True)
      
    if uploaded_files.get('superliga_players') is not None:
        player_data = pd.read_csv(uploaded_files['superliga_players'])

        player_metric_mapping = {
            'total.goals': 'Goals', 'total.assists': 'Assists', 'total.shots': 'Shots', 'total.duelsWon': 'Duels Won',
            'total.defensiveDuelsWon': 'Defensive Duels Won', 'total.passes': 'Passes', 'total.successfulPasses': 'Successful Passes',
            'total.successfulSmartPasses': 'Successful Smart Passes', 'total.successfulPassesToFinalThird': 'Successful Passes to Final Third',
            'total.successfulCrosses': 'Successful Crosses', 'total.successfulForwardPasses': 'Successful Forward Passes',
            'total.successfulBackPasses': 'Successful Back Passes', 'total.successfulThroughPasses': 'Successful Through Passes',
            'total.successfulKeyPasses': 'Successful Key Passes', 'total.successfulVerticalPasses': 'Successful Vertical Passes',
            'total.successfulLongPasses': 'Successful Long Passes', 'total.successfulDribbles': 'Successful Dribbles',
            'total.interceptions': 'Interceptions', 'total.successfulDefensiveAction': 'Successful Defensive Actions',
            'total.successfulAttackingActions': 'Successful Attacking Actions', 'total.accelerations': 'Accelerations',
            'total.pressingDuelsWon': 'Pressing Duels Won', 'total.recoveries': 'Recoveries', 'total.xgShot': 'xG Shot',
            'total.receivedPass': 'Received Passes', 'total.progressiveRun': 'Progressive Runs', 'total.touchInBox': 'Touches in Box',
            'total.successfulProgressivePasses': 'Successful Progressive Passes', 'percent.duelsWon': 'Duels Won %',
            'percent.defensiveDuelsWon': 'Defensive Duels Won %', 'percent.successfulPasses': 'Pass Accuracy %',
            'percent.successfulSmartPasses': 'Successful Smart Passes %', 'percent.successfulPassesToFinalThird': 'Successful Passes to Final Third %',
            'percent.successfulCrosses': 'Successful Crosses %', 'percent.successfulDribbles': 'Successful Dribbles %',
            'percent.shotsOnTarget': 'Shots on Target %', 'percent.successfulForwardPasses': 'Successful Forward Passes %',
            'percent.successfulThroughPasses': 'Successful Through Passes %', 'percent.successfulKeyPasses': 'Successful Key Passes %',
            'percent.successfulVerticalPasses': 'Successful Vertical Passes %', 'percent.successfulLongPasses': 'Successful Long Passes %',
            'percent.successfulProgressivePasses': 'Successful Progressive Passes %', 'percent.successfulSlidingTackles': 'Successful Sliding Tackles %'
        }

        base_cols_mapping = {
            'shortName': 'Player', 'teams.name': 'Team', 'generic.position.code': 'Position',
            'total.matches': 'Matches Played', 'total.minutesOnField': 'Minutes Played'
        }
        
        analysis_df = pd.DataFrame()
        for raw_col, pretty_col in base_cols_mapping.items():
            if raw_col in player_data.columns:
                analysis_df[pretty_col] = player_data[raw_col]
        
        for raw, pretty in player_metric_mapping.items():
            if raw in player_data.columns:
                analysis_df[pretty] = player_data[raw]

        st.sidebar.header("Player Filters")
        min_matches = st.sidebar.slider("Matches Played:", 0, int(analysis_df['Matches Played'].max()), 5)
        min_minutes = st.sidebar.slider("Minutes Played:", 0, int(analysis_df['Minutes Played'].max()), 900)
          
        available_positions = analysis_df['Position'].dropna().unique().tolist()
        selected_positions = st.sidebar.multiselect("Filter by Position:", available_positions, default=available_positions)
          
        available_teams = analysis_df['Team'].dropna().unique().tolist()
        selected_teams = st.sidebar.multiselect("Filter by Team/Club:", available_teams, default=available_teams)
        
        filtered_df = analysis_df[
            (analysis_df['Matches Played'] >= min_matches) &
            (analysis_df['Minutes Played'] >= min_minutes) &
            (analysis_df['Position'].isin(selected_positions)) &
            (analysis_df['Team'].isin(selected_teams))
        ].copy()

        tool_selection = st.radio("Select Analysis Tool:", ('Scouting Scatter Plot', 'Player Percentile Benchmark', 'Head-to-Head Comparison'))
        selectable_metrics = filtered_df.columns[5:].tolist()

        if tool_selection == 'Scouting Scatter Plot':
            st.subheader("Scouting Scatter Plot")
            if len(selectable_metrics) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_metric = st.selectbox("Select X-axis metric", selectable_metrics, key="player_x")
                with col2:
                    y_metric = st.selectbox("Select Y-axis metric", selectable_metrics, index=1, key="player_y")
                  
                fig = create_scatter_plot(filtered_df, x_metric, y_metric, f"Player Analysis: {x_metric} vs {y_metric}", text_column='Player', color_column='Team')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plotly(fig, f"player_scatter_{x_metric}_vs_{y_metric}"), unsafe_allow_html=True)
            else:
                st.warning("Not enough numeric data to create a scatter plot with the current filters.")

        elif tool_selection == 'Player Percentile Benchmark':
            st.subheader("Player Percentile Benchmark")
            player_list = filtered_df['Player'].unique().tolist()
            if not player_list:
                st.warning("No players match the current filters.")
            else:
                selected_player = st.selectbox("Select a Player:", player_list)
                params = st.multiselect("Select metrics for Benchmark Chart:", selectable_metrics, default=selectable_metrics[:8])

                if not params:
                    st.warning("Please select at least one metric.")
                else:
                    player_row = filtered_df[filtered_df['Player'] == selected_player].iloc[0]
                    position_df = filtered_df[filtered_df['Position'] == player_row['Position']]
                      
                    benchmark_data = []
                    for metric in params:
                        p90_val = (player_row[metric] / player_row['Minutes Played']) * 90 if '%' not in metric else player_row[metric]
                        percentile = int(round(percentileofscore(position_df[metric].fillna(0), player_row[metric])))
                        benchmark_data.append({'metric': metric, 'value': p90_val, 'percentile': percentile})
                      
                    benchmark_df = pd.DataFrame(benchmark_data)
                    fig = create_benchmark_chart(selected_player, benchmark_df)
                    st.pyplot(fig)
                    st.markdown(download_plot(fig, f"benchmark_chart_{selected_player}"), unsafe_allow_html=True)

        elif tool_selection == 'Head-to-Head Comparison':
            st.subheader("Head-to-Head Comparison")
            player_list = filtered_df['Player'].unique().tolist()
            if len(player_list) < 2:
                st.warning("You need at least two players matching the filters to make a comparison.")
            else:
                params = st.multiselect("Select metrics for Comparison:", selectable_metrics, default=selectable_metrics[:8])
                  
                if not params:
                    st.warning("Please select at least one metric for the comparison.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        player_a_name = st.selectbox("Select Player A:", player_list, key="player_a")
                    with col2:
                        player_b_name = st.selectbox("Select Player B:", player_list, key="player_b", index=1)

                    if player_a_name == player_b_name:
                        st.error("Please select two different players.")
                    else:
                        player_a_row = filtered_df[filtered_df['Player'] == player_a_name].iloc[0]
                        player_b_row = filtered_df[filtered_df['Player'] == player_b_name].iloc[0]
                        position_df = filtered_df[filtered_df['Position'] == player_a_row['Position']]
                          
                        benchmark_data_a = []
                        for metric in params:
                            p90_val = (player_a_row[metric] / player_a_row['Minutes Played']) * 90 if '%' not in metric else player_a_row[metric]
                            percentile = int(round(percentileofscore(position_df[metric].fillna(0), player_a_row[metric])))
                            benchmark_data_a.append({'metric': metric, 'value': p90_val, 'percentile': percentile})
                        benchmark_df_a = pd.DataFrame(benchmark_data_a)

                        benchmark_data_b = []
                        for metric in params:
                            p90_val = (player_b_row[metric] / player_b_row['Minutes Played']) * 90 if '%' not in metric else player_b_row[metric]
                            percentile = int(round(percentileofscore(position_df[metric].fillna(0), player_b_row[metric])))
                            benchmark_data_b.append({'metric': metric, 'value': p90_val, 'percentile': percentile})
                        benchmark_df_b = pd.DataFrame(benchmark_data_b)
                          
                        with col1:
                            fig_a = create_benchmark_chart(player_a_name, benchmark_df_a)
                            st.pyplot(fig_a)
                            st.markdown(download_plot(fig_a, f"benchmark_chart_{player_a_name}"), unsafe_allow_html=True)
                        with col2:
                            fig_b = create_benchmark_chart(player_b_name, benchmark_df_b)
                            st.pyplot(fig_b)
                            st.markdown(download_plot(fig_b, f"benchmark_chart_{player_b_name}"), unsafe_allow_html=True)
    else:
        st.info("Please upload a Superliga player statistics CSV file to begin analysis.")

elif analysis_type == "Pizza Plots":
    st.markdown('<h2 class="sub-header">Player Pizza Plot Generator</h2>', unsafe_allow_html=True)
    st.info("ℹ️ This tool uses the 'Superliga Player Stats' file. Please ensure it's uploaded.")
      
    if uploaded_files.get('superliga_players') is not None:
        player_data = pd.read_csv(uploaded_files.get('superliga_players'))
          
        plot_mode = st.radio("Select Plot Mode:", ('Default Template', 'Custom Plot'))

        master_metric_mapping = {
            'shortName': 'Player', 'teams.name': 'Team', 'generic.position.code': 'Position',
            'birthDate': 'Birth Date', 'total.minutesOnField': 'Minutes Played',
            'total.goals': 'Goals', 'total.assists': 'Assists', 'total.shots': 'Shots', 'total.duelsWon': 'Duels Won',
            'total.defensiveDuelsWon': 'Defensive Duels Won', 'total.passes': 'Passes', 'total.successfulPasses': 'Successful Passes',
            'total.successfulSmartPasses': 'Successful Smart Passes', 'total.successfulPassesToFinalThird': 'Successful Passes to Final Third',
            'total.successfulCrosses': 'Successful Crosses', 'total.successfulForwardPasses': 'Successful Forward Passes',
            'total.successfulBackPasses': 'Successful Back Passes', 'total.successfulThroughPasses': 'Successful Through Passes',
            'total.successfulKeyPasses': 'Successful Key Passes', 'total.successfulVerticalPasses': 'Successful Vertical Passes',
            'total.successfulLongPasses': 'Successful Long Passes', 'total.successfulDribbles': 'Successful Dribbles',
            'total.interceptions': 'Interceptions', 'total.successfulDefensiveAction': 'Successful Defensive Actions',
            'total.successfulAttackingActions': 'Successful Attacking Actions', 'total.accelerations': 'Accelerations',
            'total.pressingDuelsWon': 'Pressing Duels Won', 'total.recoveries': 'Recoveries', 'total.xgShot': 'xG Shot',
            'total.receivedPass': 'Received Passes', 'total.progressiveRun': 'Progressive Runs', 'total.touchInBox': 'Touches in Box',
            'total.successfulProgressivePasses': 'Successful Progressive Passes', 'percent.duelsWon': 'Duels Won %',
            'percent.defensiveDuelsWon': 'Defensive Duels Won %', 'percent.successfulPasses': 'Pass Accuracy %',
            'percent.successfulSmartPasses': 'Successful Smart Passes %', 'percent.successfulPassesToFinalThird': 'Successful Passes to Final Third %',
            'percent.successfulCrosses': 'Successful Crosses %', 'percent.successfulDribbles': 'Successful Dribbles %',
            'percent.shotsOnTarget': 'Shots on Target %', 'percent.successfulForwardPasses': 'Successful Forward Passes %',
            'percent.successfulThroughPasses': 'Successful Through Passes %', 'percent.successfulKeyPasses': 'Successful Key Passes %',
            'percent.successfulVerticalPasses': 'Successful Vertical Passes %', 'percent.successfulLongPasses': 'Successful Long Passes %',
            'percent.successfulProgressivePasses': 'Successful Progressive Passes %', 'total.slidingTackles': 'Tackles'
        }

        df_cleaned = player_data.copy()
        df_cleaned.rename(columns=master_metric_mapping, inplace=True)

        if plot_mode == 'Default Template':
            params_att = ['Goals', 'Shots', 'Shots on Target %', 'xG Shot', 'Successful Dribbles', 'Progressive Runs']
            params_poss = ['Assists', 'Successful Passes', 'Pass Accuracy %', 'Successful Progressive Passes']
            params_def = ['Tackles', 'Interceptions', 'Duels Won %', 'Recoveries']
            params = params_att + params_poss + params_def
              
            slice_colors = []
            for p in params:
                if p in params_att: slice_colors.append('#FE6100')
                elif p in params_poss: slice_colors.append('#648FFF')
                elif p in params_def: slice_colors.append('#FFB000')
                else: slice_colors.append('#EBEBE9')
              
            show_legend_flag = True
              
        else: # Custom Plot mode
            all_pretty_metrics = [v for k, v in master_metric_mapping.items() if k not in ['shortName', 'teams.name', 'generic.position.code', 'birthDate', 'total.minutesOnField']]
            params = st.multiselect("Select metrics for Custom Pizza Plot:", all_pretty_metrics, default=all_pretty_metrics[:5])
            slice_colors = '#648FFF'
            show_legend_flag = False
          
        required_metrics = [col for col in params if col not in df_cleaned.columns]
        if any(metric in df_cleaned.columns for metric in required_metrics):
             st.error(f"The uploaded player file is missing required columns for the pizza plot: {', '.join(required_metrics)}. Please check the file.")
        else:
            min_minutes = st.slider("Filter players by minimum minutes played:", 0, int(df_cleaned['Minutes Played'].max()), 900)
              
            available_teams = sorted(df_cleaned['Team'].dropna().unique().tolist())
            selected_teams = st.multiselect("Filter by Team/Club:", available_teams, default=available_teams)
              
            filtered_players_df = df_cleaned[
                (df_cleaned['Minutes Played'] >= min_minutes) &
                (df_cleaned['Team'].isin(selected_teams))
            ].copy()
              
            player_list = sorted(filtered_players_df['Player'].unique().tolist())
            if not player_list:
                st.warning("No players match the current filters.")
            else:
                selected_player = st.selectbox("Select player:", player_list)
                  
                if not params:
                    st.warning("Please select at least one metric for the custom plot.")
                else:
                    params_p90 = []
                    for p in params:
                        if '%' not in p and p != 'Minutes Played':
                            param_p90_name = f"{p} p90"
                            if p in filtered_players_df.columns and 'Minutes Played' in filtered_players_df.columns:
                                filtered_players_df[param_p90_name] = (filtered_players_df[p] / filtered_players_df['Minutes Played']) * 90
                            params_p90.append(param_p90_name)
                        else:
                            params_p90.append(p)
                      
                    player_info_row = filtered_players_df[filtered_players_df['Player'] == selected_player].iloc[0]
                      
                    try:
                        birth_date = pd.to_datetime(player_info_row['Birth Date'])
                        age = datetime.now().year - birth_date.year - ((datetime.now().month, datetime.now().day) < (birth_date.month, birth_date.day))
                    except:
                        age = "N/A"

                    player_info = {'name': selected_player, 'age': age, 'team': player_info_row['Team'], 'position': player_info_row['Position']}

                    percentile_values = [int(round(percentileofscore(filtered_players_df[p].fillna(0), player_info_row[p]))) for p in params_p90]
                      
                    fig = create_pizza_plot(
                        percentile_values,
                        params_p90,
                        player_info,
                        slice_colors,
                        param_colors='#EBEBE9',
                        show_legend=show_legend_flag
                    )
                    st.pyplot(fig)
                    st.markdown(download_plot(fig, f"pizza_plot_{selected_player}"), unsafe_allow_html=True)
    else:
        st.info("Please upload a Superliga player statistics CSV file to create pizza plots.")

elif analysis_type in ["Event Data Analysis", "Pass Maps", "Shot Maps", "Pass Networks", "Heatmaps", "Physical Analysis"]:
    st.markdown(f'<h2 class="sub-header">{analysis_type}</h2>', unsafe_allow_html=True)
          
    if analysis_type == "Physical Analysis":
        if uploaded_files.get('physical_stats') is not None:
            physical_data = pd.read_csv(uploaded_files['physical_stats'])
            st.subheader("Physical Dataset Overview")
            st.dataframe(physical_data.head())

            match_labels = {}
            for game_id in physical_data['game_id'].unique():
                teams = physical_data[physical_data['game_id'] == game_id]['team_name'].unique()
                match_labels[game_id] = f"{teams[0]} vs {teams[1]}" if len(teams) == 2 else f"Match ID: {game_id}"

            selected_label = st.selectbox("Select a match to analyze", list(match_labels.values()))
            selected_game_id = [gid for gid, label in match_labels.items() if label == selected_label][0]
            match_data = physical_data[physical_data['game_id'] == selected_game_id].copy()
              
            physical_analysis_type = st.radio("Select Analysis Type:", ('Top Performers (Table)', 'Metric Comparison (Scatter)', 'Top 10 Ranking (Bar Chart)'))
            numeric_cols = match_data.select_dtypes(include=np.number).columns.tolist()
            metrics_to_exclude = ['player_id', 'game_id', 'team_id', 'repetition']
            selectable_metrics = [col for col in numeric_cols if col not in metrics_to_exclude]

            if physical_analysis_type == 'Top Performers (Table)':
                metric_for_ranking = st.selectbox("Select metric for ranking", selectable_metrics)
                top_n = st.slider("Number of top players to show", 5, 20, 10)
                display_data = match_data.nlargest(top_n, metric_for_ranking)
                  
                for col in display_data.columns:
                    if 'Distance' in col:
                        display_data[col] = display_data[col].round(0)
                        display_data.rename(columns={col: f"{col} (m)"}, inplace=True)
                if 'Max Speed' in display_data.columns:
                    display_data.rename(columns={'Max Speed': 'Max Speed (km/h)'}, inplace=True)

                st.subheader(f"Top {top_n} performers for {metric_for_ranking}")
                display_column = f"{metric_for_ranking} (m)" if "Distance" in metric_for_ranking else ("Max Speed (km/h)" if "Max Speed" in metric_for_ranking else metric_for_ranking)
                st.dataframe(display_data[['player_name', 'team_name', display_column]])

            elif physical_analysis_type == 'Metric Comparison (Scatter)':
                col1, col2 = st.columns(2)
                with col1:
                    x_metric = st.selectbox("Select X-axis metric", selectable_metrics, key="phys_x")
                with col2:
                    y_metric = st.selectbox("Select Y-axis metric", selectable_metrics, index=1, key="phys_y")
                fig = create_scatter_plot(match_data, x_metric, y_metric, f"Physical Analysis: {x_metric} vs {y_metric}", text_column='player_name', color_column='team_name')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plotly(fig, f"physical_scatter_{x_metric}_vs_{y_metric}"), unsafe_allow_html=True)

            elif physical_analysis_type == 'Top 10 Ranking (Bar Chart)':
                metric_for_ranking = st.selectbox("Select metric for ranking", selectable_metrics)
                top_10 = match_data.nlargest(10, metric_for_ranking)
                fig = px.bar(top_10, x=metric_for_ranking, y='player_name', orientation='h',
                             title=f"Top 10 Players for {metric_for_ranking}", color='team_name',
                             color_discrete_map={team: color for team, color in zip(top_10['team_name'].unique(), ['#648FFF', '#DC267F'])})
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plotly(fig, f"physical_barchart_{metric_for_ranking}"), unsafe_allow_html=True)
        else:
            st.info("Please upload a physical metrics CSV file to begin analysis.")
          
    elif uploaded_files.get('event_data') is not None:
        try:
            event_data = pd.read_csv(uploaded_files['event_data'])
            st.success("Event data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
              
        if analysis_type == "Event Data Analysis":
            st.subheader("Comprehensive Match Analysis for Coaches")
            
            # Team selection for analysis
            if 'team.name' in event_data.columns and 'matchId' in event_data.columns:
                
                # Create user-friendly match selection interface
                match_labels = {}
                for match_id in event_data['matchId'].unique():
                    teams = event_data[event_data['matchId'] == match_id]['team.name'].unique()
                    match_labels[match_id] = f"{teams[0]} vs {teams[1]}" if len(teams) == 2 else f"Match ID: {match_id}"

                selected_label = st.selectbox("First, select a match to analyze", list(match_labels.values()))
                selected_match_id = [mid for mid, label in match_labels.items() if label == selected_label][0]
                match_df = event_data[event_data['matchId'] == selected_match_id].copy()
                  
                teams_in_match = match_df['team.name'].dropna().unique()
                if len(teams_in_match) == 0:
                    st.warning("No teams found for the selected match.")
                    st.stop()
                    
                selected_team_for_analysis = st.selectbox("Select team for detailed analysis:", teams_in_match)
                team_events = match_df[match_df['team.name'] == selected_team_for_analysis].copy()
                
                # Analysis type selection with new options
                event_analysis_type = st.radio("Select Analysis Type:", 
                    ('Detailed Shot Analysis', 'Possession Flow Analysis', 'Match Flow Analysis', 'Action Success by Zones', 'Defensive Actions Analysis', 'Passing Network Analysis', 'Set Piece Analysis'))
                
                # --- NEW: Detailed Shot Analysis ---
                if event_analysis_type == 'Detailed Shot Analysis':
                    st.subheader(f"Detailed Shot Analysis - {selected_team_for_analysis}")
                    st.info("🎯 This dashboard breaks down every shot, showing location, body part used, goal placement, and expected goal values.")
                    
                    shot_events = team_events[team_events['type.primary'] == 'shot'].copy()
                    
                    # Display key metrics
                    total_shots = len(shot_events)
                    goals = shot_events['shot.isGoal'].sum()
                    total_xg = shot_events['shot.xg'].sum()
                    total_psxg = shot_events['shot.postShotXg'].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Shots", total_shots)
                    col2.metric("Goals", int(goals))
                    col3.metric("Total xG", f"{total_xg:.2f}")
                    col4.metric("Total post-shot xG", f"{total_psxg:.2f}")

                    fig_shot, shot_charts = create_detailed_shot_map(shot_events, selected_team_for_analysis)
                    
                    if fig_shot:
                        st.pyplot(fig_shot)
                        st.markdown(download_plot(fig_shot, f"detailed_shot_map_{selected_team_for_analysis}"), unsafe_allow_html=True)
                    
                    if shot_charts:
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'body_part' in shot_charts:
                                st.plotly_chart(shot_charts['body_part'], use_container_width=True)
                        with col2:
                            if 'goal_zone' in shot_charts:
                                st.plotly_chart(shot_charts['goal_zone'], use_container_width=True)

                # --- NEW: Possession Flow Analysis ---
                elif event_analysis_type == 'Possession Flow Analysis':
                    st.subheader(f"Possession Flow Analysis - {selected_team_for_analysis}")
                    st.info("🌊 This visualization tracks possession sequences from start to finish, highlighting which attacks lead to shots and goals.")

                    fig_poss, poss_charts = create_possession_flow_map(team_events, selected_team_for_analysis)

                    if fig_poss:
                        st.pyplot(fig_poss)
                        st.markdown(download_plot(fig_poss, f"possession_flow_{selected_team_for_analysis}"), unsafe_allow_html=True)
                    
                    if poss_charts and 'flank_analysis' in poss_charts:
                        st.plotly_chart(poss_charts['flank_analysis'], use_container_width=True)

                # --- EXISTING ANALYSIS MODULES ---
                elif event_analysis_type == 'Match Flow Analysis':
                    st.subheader(f"Match Flow Analysis - {selected_team_for_analysis}")
                    st.info("📈 This analysis shows how your team's passing accuracy and defensive interceptions evolve throughout the match.")
                    
                    if 'minute' in team_events.columns:
                        team_events['time_bin'] = (team_events['minute'] // 10) * 10
                        flow_metrics = []
                        for time_bin in sorted(team_events['time_bin'].unique()):
                            bin_events = team_events[team_events['time_bin'] == time_bin]
                            passes = bin_events[bin_events['type.primary'] == 'pass']
                            shots = bin_events[bin_events['type.primary'] == 'shot']
                            # --- NEW: Calculate Interceptions ---
                            interceptions = bin_events[bin_events['type.primary'] == 'interception']
                            
                            metrics = {
                                'Time Period': f"{int(time_bin)}-{int(time_bin+10)} min",
                                'Pass Accuracy %': (passes['pass.accurate'].sum() / len(passes) * 100) if len(passes) > 0 else 0,
                                'Total Actions': len(bin_events),
                                'Shots Taken': len(shots),
                                'Interceptions': len(interceptions) # Add interception count
                            }
                            flow_metrics.append(metrics)
                        
                        flow_df = pd.DataFrame(flow_metrics)
                        
                        # --- NEW: Create Dual-Axis Chart ---
                        from plotly.subplots import make_subplots
                        import plotly.graph_objects as go

                        fig_flow = make_subplots(specs=[[{"secondary_y": True}]])

                        # Add Pass Accuracy trace
                        fig_flow.add_trace(
                            go.Scatter(x=flow_df['Time Period'], y=flow_df['Pass Accuracy %'], name="Pass Accuracy %", line=dict(color='#648FFF')),
                            secondary_y=False,
                        )

                        # Add Interceptions trace
                        fig_flow.add_trace(
                            go.Bar(x=flow_df['Time Period'], y=flow_df['Interceptions'], name="Interceptions", marker_color='#DC267F', opacity=0.6),
                            secondary_y=True,
                        )

                        # Update layout and axis titles
                        fig_flow.update_layout(
                            title_text=f"Match Flow: Passing vs. Interceptions for {selected_team_for_analysis}",
                            template='plotly_dark'
                        )
                        fig_flow.update_xaxes(title_text="Time Period")
                        fig_flow.update_yaxes(title_text="<b>Pass Accuracy %</b>", secondary_y=False)
                        fig_flow.update_yaxes(title_text="<b>Number of Interceptions</b>", secondary_y=True)
                        
                        st.plotly_chart(fig_flow, use_container_width=True)
                        st.markdown(download_plotly(fig_flow, f"match_flow_{selected_team_for_analysis}"), unsafe_allow_html=True)
                        
                        st.subheader("Detailed Time Period Breakdown")
                        st.dataframe(flow_df)
                        
                        with st.expander("💡 Coaching Insights from Match Flow"):
                            st.write("""
                            **How to interpret this analysis:**
                            - **Pass Accuracy Trends:** Look for periods where accuracy drops significantly. This often indicates fatigue, pressure, or tactical adjustments by the opponent.
                            - **Interceptions:** Spikes in interceptions can indicate periods of effective defensive pressure or moments where the opponent was trying to force risky passes.
                            - **Training Application:** Use these insights to identify when your team's concentration might be dropping or when your defensive press is most effective.
                            """)
                    else:
                        st.warning("Time data (minute column) not available in this dataset for match flow analysis.")

                elif event_analysis_type == 'Action Success by Zones':
                    st.subheader(f"Action Success by Field Zones - {selected_team_for_analysis}")
                    st.info("🗺️ This analysis reveals where on the pitch your team is most effective, helping optimize positional play and identify weaknesses.")
                    
                    if 'location.x' in team_events.columns and 'location.y' in team_events.columns:
                        action_type = st.selectbox("Select action type to analyze:", ['pass', 'shot', 'duel', 'cross', 'dribble'])
                        action_events = team_events[team_events['type.primary'] == action_type]
                        
                        if not action_events.empty:
                            st.subheader(f"Activity Density Heatmap: {action_type.title()}s")
                            pitch_density = Pitch(pitch_type='wyscout', pitch_color='#22312b', line_color='#efefef', line_zorder=2)
                            fig_density, ax_density = pitch_density.draw(figsize=(14, 10))
                            fig_density.set_facecolor('#22312b')
                            
                            bin_statistic = pitch_density.bin_statistic(action_events['location.x'], action_events['location.y'], statistic='count', bins=(12, 8))
                            bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
                            pcm = pitch_density.heatmap(bin_statistic, ax=ax_density, cmap='hot', edgecolors='#22312b')
                            
                            cbar = fig_density.colorbar(pcm, ax=ax_density, shrink=0.6)
                            cbar.outline.set_edgecolor('#efefef')
                            cbar.ax.yaxis.set_tick_params(color='#efefef')
                            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
                            cbar.set_label('Action Frequency', color='#efefef', fontsize=12)
                            
                            ax_density.set_title(f'{action_type.title()} Density Map - {selected_team_for_analysis}', fontsize=16, pad=20, color='white')
                            st.pyplot(fig_density)
                            st.markdown(download_plot(fig_density, f"density_map_{action_type}_{selected_team_for_analysis}"), unsafe_allow_html=True)
                        else:
                            st.warning(f"No {action_type} events found for {selected_team_for_analysis} in this dataset.")
                    else:
                        st.warning("Location data not available in this dataset for spatial analysis.")

                elif event_analysis_type == 'Defensive Actions Analysis':
                    st.subheader(f"Defensive Actions Analysis - {selected_team_for_analysis}")
                    st.info("🛡️ This analysis examines your team's defensive effectiveness by mapping where key actions occur on the pitch.")
                    
                    # --- Define and filter for successful defensive actions ---
                    # We consider interceptions, clearances, and blocks as inherently successful.
                    # For tackles, we infer success if the team keeps possession on the next event.
                    
                    # Sort events to accurately determine the next action
                    team_events_sorted = team_events.sort_values(by=['minute', 'second']).reset_index(drop=True)
                    team_events_sorted['next_event_team'] = team_events_sorted['team.name'].shift(-1)
                    
                    # Filter for successful tackles
                    successful_tackles = team_events_sorted[
                        (team_events_sorted['type.primary'] == 'tackle') &
                        (team_events_sorted['team.name'] == team_events_sorted['next_event_team'])
                    ]
                    
                    # Filter for other inherently successful actions
                    other_successful_actions = team_events_sorted[
                        team_events_sorted['type.primary'].isin(['interception', 'clearance', 'block'])
                    ]
                    
                    # Combine into a single DataFrame
                    successful_defensive_events = pd.concat([successful_tackles, other_successful_actions])
                    
                    if successful_defensive_events.empty:
                        st.warning("No successful defensive actions could be identified in this match's data.")
                    else:
                        # --- UI to select visualization type ---
                        viz_type = st.radio(
                            "Select Visualization Type:",
                            ("Successful Actions Map", "Key Defensive Areas (Heatmap)")
                        )

                        fig = None
                        if viz_type == "Successful Actions Map":
                            fig = create_defensive_actions_map(successful_defensive_events, selected_team_for_analysis)
                        
                        elif viz_type == "Key Defensive Areas (Heatmap)":
                            fig = create_defensive_heatmap(successful_defensive_events, selected_team_for_analysis)

                        if fig:
                            st.pyplot(fig)
                            st.markdown(download_plot(fig, f"defensive_analysis_{selected_team_for_analysis}"), unsafe_allow_html=True)
                elif event_analysis_type == 'Passing Network Analysis':
                    st.subheader(f"Team Passing Patterns - {selected_team_for_analysis}")
                    st.info("🔗 This analysis reveals how your team moves the ball, showing passing relationships and identifying key playmakers.")
                    
                    passes = team_events[team_events['type.primary'] == 'pass']
                    if not passes.empty and 'player.name' in passes.columns and 'pass.recipient.name' in passes.columns:
                        pass_combinations = passes.groupby(['player.name', 'pass.recipient.name']).size().reset_index(name='Pass Count')
                        pass_combinations = pass_combinations.sort_values('Pass Count', ascending=False).head(15)
                        st.subheader("Most Frequent Passing Combinations")
                        st.dataframe(pass_combinations)
                    else:
                        st.warning("Insufficient passing data available for network analysis.")

                elif event_analysis_type == 'Set Piece Analysis':
                    st.subheader(f"Set Piece Analysis - {selected_team_for_analysis}")
                    st.info("⚽ This analysis examines your team's effectiveness in set piece situations.")
                    
                    set_piece_mask = team_events['type.primary'].isin(['free_kick', 'corner', 'throw_in', 'penalty'])
                    if 'type.secondary' in team_events.columns:
                        set_piece_mask |= team_events['type.secondary'].isin(['free_kick', 'corner', 'throw_in', 'penalty'])
                    set_pieces = team_events[set_piece_mask]

                    if not set_pieces.empty:
                        set_piece_types = set_pieces['type.primary'].value_counts()
                        fig_set_pieces = px.bar(x=set_piece_types.index, y=set_piece_types.values, title=f"Set Piece Distribution - {selected_team_for_analysis}")
                        st.plotly_chart(fig_set_pieces, use_container_width=True)
                    else:
                        st.warning("No set piece data found in this dataset.")
            else:
                st.warning("The event data must contain 'matchId' and 'team.name' columns.")

        elif analysis_type in ["Pass Maps", "Shot Maps", "Pass Networks", "Heatmaps"]:
            if 'matchId' in event_data.columns and 'team.name' in event_data.columns:
                match_labels = {}
                for match_id in event_data['matchId'].unique():
                    teams = event_data[event_data['matchId'] == match_id]['team.name'].unique()
                    match_labels[match_id] = f"{teams[0]} vs {teams[1]}" if len(teams) == 2 else f"Match ID: {match_id}"

                selected_label = st.selectbox("First, select a match to analyze", list(match_labels.values()))
                selected_match_id = [mid for mid, label in match_labels.items() if label == selected_label][0]
                match_df = event_data[event_data['matchId'] == selected_match_id].copy()
                  
                teams_in_match = match_df['team.name'].dropna().unique()
                if len(teams_in_match) > 0:
                    selected_team = st.selectbox("Now, select the team", teams_in_match)
                    team_match_df = match_df[match_df['team.name'] == selected_team]
                          
                    fig = None
                    if analysis_type == "Pass Maps":
                        col1, col2 = st.columns(2)
                        with col1:
                            location_filter_option = st.selectbox("Filter passes by location", ['All Passes', 'Forward Passes', 'Final Third Passes', 'Own Third Passes'])
                        with col2:
                            pass_filter_option = st.selectbox("Filter passes by success", ['All Passes', 'Successful Only', 'Unsuccessful Only'])
                        st.subheader(f"Displaying: {location_filter_option} for {selected_team}")
                        fig = create_pass_map(team_match_df, selected_team, pass_filter_option, location_filter_option)

                    elif analysis_type == "Shot Maps":
                        st.subheader(f"Displaying shots for {selected_team}")
                        fig = create_shot_map(team_match_df, selected_team)

                    elif analysis_type == "Pass Networks":
                        st.subheader(f"Displaying Pass Network for {selected_team}")
                        fig = create_pass_network(team_match_df, selected_team)

                    elif analysis_type == "Heatmaps":
                        st.subheader(f"Displaying Heatmap for {selected_team}")
                        fig = create_heatmap(team_match_df, selected_team)

                    if fig is not None:
                        st.pyplot(fig)
                        st.markdown(download_plot(fig, f"{analysis_type.lower().replace(' ', '_')}_{selected_team}"), unsafe_allow_html=True)
            else:
                st.warning("The event data must contain 'matchId' and 'team.name' columns.")
    else:
        st.info("Please upload an event data CSV file to begin analysis.")
        