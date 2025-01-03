import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.cluster import DBSCAN
from datetime import timedelta
from geopy.distance import geodesic
from tqdm import tqdm

matplotlib.use("module://matplotlib_inline.backend_inline")  # Keeps the backend interactive
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # Use pdflatex for LaTeX integration
    "font.family": "serif",  # Use a serif font matching LaTeX
    "text.usetex": True,  # Use TeX for text rendering
    "pgf.rcfonts": False,  # Prevent overriding LaTeX fonts
})

def plot_matches_interactive_map(matches_df, show_lines=True, save_html=None):
    """
    Create an interactive map showing MODIS and VIIRS detections with their matches.

    Parameters:
    matches_df: DataFrame with columns for both MODIS and VIIRS coordinates
    show_lines: bool, whether to draw lines between matched detections
    save_html: str or None, path to save the map as HTML file

    Returns:
    folium.Map object
    """
    # Calculate center of Germany
    center_lat = (matches_df['modis_lat'].mean() + matches_df['viirs_lat'].mean()) / 2
    center_lon = (matches_df['modis_lon'].mean() + matches_df['viirs_lon'].mean()) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'  # Light theme map
    )

    # Create feature groups for different layers
    modis_group = folium.FeatureGroup(name='MODIS Detections')
    viirs_group = folium.FeatureGroup(name='VIIRS Detections')
    lines_group = folium.FeatureGroup(name='Matches')

    # Add markers and lines
    for _, match in matches_df.iterrows():
        # MODIS marker (red)
        modis_popup = f"""
            MODIS Detection<br>
            Time: {match['modis_time']}<br>
            Confidence: {match.get('modis_confidence', 'N/A')}<br>
            Brightness: {match.get('modis_brightness', 'N/A')}<br>
            Time Difference: {match['time_diff_minutes']:.1f} min
        """
        folium.CircleMarker(
            location=[match['modis_lat'], match['modis_lon']],
            radius=6,
            color='red',
            fill=True,
            popup=modis_popup,
            weight=2
        ).add_to(modis_group)

        # VIIRS marker (blue)
        viirs_popup = f"""
            VIIRS Detection<br>
            Time: {match['viirs_time']}<br>
            Distance: {match['distance_km']:.1f} km
        """
        folium.CircleMarker(
            location=[match['viirs_lat'], match['viirs_lon']],
            radius=6,
            color='blue',
            fill=True,
            popup=viirs_popup,
            weight=2
        ).add_to(viirs_group)

        # Line between matches
        if show_lines:
            folium.PolyLine(
                locations=[
                    [match['modis_lat'], match['modis_lon']],
                    [match['viirs_lat'], match['viirs_lon']]
                ],
                color='gray',
                weight=1,
                opacity=0.5
            ).add_to(lines_group)

    # Add all layers to map
    modis_group.add_to(m)
    viirs_group.add_to(m)
    if show_lines:
        lines_group.add_to(m)

    folium.LayerControl().add_to(m)

    # Add a title
    title_html = '''
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 300px; z-index:9999;
                    background-color: white; padding: 10px; border-radius: 5px;">
            <h4>Fire Detections: MODIS vs VIIRS</h4>
            <p style="font-size: 12px;">
                Red: MODIS detections<br>
                Blue: VIIRS detections<br>
                Gray lines: Matched pairs
            </p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save if requested
    if save_html:
        m.save(save_html)

    return m

def create_map_screenshot(matches_df, center_coords, zoom_level=13, width=800, height=600,
                         output_file='map_screenshot.png', show_lines=True):
    """
    Create a static screenshot of a map centered on specific coordinates.

    Parameters:
    matches_df: DataFrame with matches (same format as in plot_matches_map)
    center_coords: tuple of (latitude, longitude) for the map center
    zoom_level: int, zoom level (higher numbers = more zoomed in)
    width: int, width of the output image in pixels
    height: int, height of the output image in pixels
    output_file: str, path where to save the PNG file
    show_lines: bool, whether to show the gray lines between matches
    """
    m = folium.Map(
        location=center_coords,
        zoom_start=zoom_level,
        tiles='CartoDB positron',
    )

    # Add markers and lines
    for _, match in matches_df.iterrows():
        # MODIS marker (red)
        folium.CircleMarker(
            location=[match['modis_lat'], match['modis_lon']],
            radius=6,
            color='red',
            fill=True,
            weight=2,
            popup=f"MODIS: {match['modis_time']:%Y-%m-%d %H:%M}"
        ).add_to(m)

        # VIIRS marker (blue)
        folium.CircleMarker(
            location=[match['viirs_lat'], match['viirs_lon']],
            radius=6,
            color='blue',
            fill=True,
            weight=2,
            popup=f"VIIRS: {match['viirs_time']:%Y-%m-%d %H:%M}"
        ).add_to(m)

        if show_lines:
            folium.PolyLine(
                locations=[
                    [match['modis_lat'], match['modis_lon']],
                    [match['viirs_lat'], match['viirs_lon']]
                ],
                color='gray',
                weight=1,
                opacity=0.5
            ).add_to(m)

    legend_html = '''
        <div style="position: absolute;
                    bottom: 10px; left: 10px;
                    z-index: 1000;
                    background-color: white;
                    padding: 6px;
                    border-radius: 4px;
                    font-size: 12px;
                    line-height: 1.5;">
            <div><span style="color: red;">●</span> MODIS detections</div>
            <div><span style="color: blue;">●</span> VIIRS detections</div>
            <div><span style="color: gray;">―</span> Matched pairs</div>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    css = """
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        #map {
            position: fixed !important;
            width: %dpx !important;
            height: %dpx !important;
        }
        .leaflet-control-container {
            display: none;
        }
    </style>
    """ % (width, height)

    html = f"""
    <!DOCTYPE html>
    <head>{css}</head>
    <body>
        <div id="map"></div>
        {m.get_root().render()}
    </body>
    </html>
    """

    temp_html = 'temp_map.html'
    with open(temp_html, 'w') as f:
        f.write(html)

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument(f'--window-size={width},{height}')

    browser = webdriver.Chrome(options=chrome_options)
    browser.get(f'file://{os.path.abspath(temp_html)}')

    # Wait for map tiles to load
    time.sleep(3)
    browser.save_screenshot(output_file)
    browser.quit()
    os.remove(temp_html)

def plot_histogram(matches, dataset_name):
    # Histogram of time differences
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=matches,
        x='time_diff_minutes',
        bins=30,
        color='blue',
        alpha=0.6
    )
    plt.xlabel(f'Time Difference (minutes)\nNegative = {dataset_name} Earlier, Positive = VIIRS Earlier')
    plt.ylabel('Count')

    # Add vertical line at 0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Display the plot and save as PGF
    plt.tight_layout()
    plt.savefig(f'latex_plots/histogram_{dataset_name.lower().replace(" ", "_")}.pgf', format="pgf")
    # dont show the title for latex plots
    plt.title(f'Distribution of Detection Time Differences ({dataset_name} vs. VIIRS)')
    plt.show()

def plot_time_distance(matches, dataset_name):
    # Time differences vs. Distance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=matches,
        x='distance_km',
        y='time_diff_minutes',
        alpha=0.5
    )
    plt.xlabel('Distance between detections (km)')
    plt.ylabel('Time Difference (minutes)')

    # display the plot and save as PGF
    plt.tight_layout()
    plt.savefig(f'latex_plots/time_distance_{dataset_name.lower().replace(" ", "_")}.pgf', format="pgf")
    # dont show the title for latex plots
    plt.title(f'Time Difference vs. Spatial Distance ({dataset_name} vs. VIIRS)')
    plt.show()


def show_screenshots(path1, path2, title1, title2):
    # display the maps
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(path1))
    plt.axis('off')
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(plt.imread(path2))
    plt.axis('off')
    plt.title(title2)

    plt.tight_layout()
    plt.show()