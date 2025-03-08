# The MIT License

# Copyright (c) 2025 Albert Murienne

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import gradio as gr
import html
import numpy as np
import folium
from fitparse import FitFile
from pykalman import KalmanFilter
from geopy.distance import geodesic

def extract_fit_data(fit_file_path):
    """Extracts lat, lon, and speed (if available) from a .fit file."""
    fitfile = FitFile(fit_file_path)
    latitudes, longitudes, speeds, timestamps = [], [], [], []

    for record in fitfile.get_messages("record"):
        lat, lon, speed, timestamp = None, None, None, None

        for data in record:

            if data.name == "position_lat":
                lat = data.value / (2**31) * 180  # Convert Garmin's semi-circle format to degrees
            elif data.name == "position_long":
                lon = data.value / (2**31) * 180
            elif data.name == "enhanced_speed":  # Speed in m/s
                speed = data.value
            elif data.name == "timestamp":
                timestamp = data.value.timestamp()  # Convert to UNIX time

        if lat is not None and lon is not None:
            latitudes.append(lat)
            longitudes.append(lon)
            timestamps.append(timestamp if timestamp is not None else 0)
            speeds.append(speed if speed is not None else np.nan)  # Use NaN for missing speeds

    return np.array(latitudes), np.array(longitudes), np.array(speeds), np.array(timestamps)

def compute_speed_from_gps(latitudes, longitudes, timestamps):
    """Estimates speed (m/s) from GPS data using Haversine distance and time."""
    estimated_speeds = np.zeros_like(latitudes, dtype=float)

    for i in range(1, len(latitudes)):
        distance = geodesic((latitudes[i-1], longitudes[i-1]), (latitudes[i], longitudes[i])).meters
        time_diff = timestamps[i] - timestamps[i-1]

        estimated_speeds[i] = distance / time_diff if time_diff > 0 else 0  # Avoid division by zero

    return estimated_speeds

def apply_kalman_filter(latitudes, longitudes, speeds, timestamps):
    """Applies a Kalman Filter to smooth lat, lon, and estimated speed."""
    if np.isnan(speeds).all():  # If all speed values are missing, estimate them
        speeds = compute_speed_from_gps(latitudes, longitudes, timestamps)
    else:
        print("using native speed measurements")

    initial_state = [latitudes[0], longitudes[0], speeds[0]]

    transition_matrix = np.eye(3)  # Identity matrix (assumes smooth movement)
    observation_matrix = np.eye(3)  # Direct observation

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        observation_covariance=np.eye(3) * 0.0001,  # Adjust GPS noise level
        transition_covariance=np.eye(3) * 0.00001,  # Adjust for smoother tracking
    )

    smoothed_states, _ = kf.smooth(np.column_stack([latitudes, longitudes, speeds]))
    return smoothed_states[:, 0], smoothed_states[:, 1], smoothed_states[:, 2]  # Smoothed lat, lon, speed

def interpolate_color(val, colors):
    """Interpolates between given colors based on val (0 to 1)."""
    val = np.clip(val, 0, 1)  # Ensure value is within range

    idx = int(val * (len(colors) - 1))  # Find lower index
    frac = (val * (len(colors) - 1)) - idx  # Compute fractional part

    # Linear interpolation between two nearest colors
    color1 = np.array(colors[idx])
    color2 = np.array(colors[min(idx + 1, len(colors) - 1)])
    interpolated = (1 - frac) * color1 + frac * color2

    return [int(c) for c in interpolated]  # Convert to RGB

def speed_to_color(speed, min_speed, max_speed):
    """Maps speed to color."""
    if max_speed == min_speed:  # Prevent division by zero
        norm_speed = 0.5
    else:
        norm_speed = (speed - min_speed) / (max_speed - min_speed)

    # Define color gradient (approximate "plasma")
    colors = [
        (13, 8, 135),  # Dark blue
        (156, 23, 158),  # Purple
        (237, 121, 83),  # Orange
        (240, 249, 33)  # Yellow
    ]

    return interpolate_color(norm_speed, colors)

def plot_colored_route(fit_file_path):
    """Plots an activity map with speed-based colors (from .fit file)."""
    latitudes, longitudes, speeds, timestamps = extract_fit_data(fit_file_path)

    print(f"plotting {fit_file_path}")

    if not latitudes.size:
        print("No GPS data found in the file.")
        return

    # Apply Kalman filter (with speed estimation if missing)
    filtered_lat, filtered_lon, filtered_speed = apply_kalman_filter(latitudes, longitudes, speeds, timestamps)

    # Handle missing or constant speed values
    if np.isnan(filtered_speed).all() or (np.max(filtered_speed) == np.min(filtered_speed)):
        print("Warning: No valid speed variation detected. Using default color.")
        filtered_speed = np.zeros_like(filtered_speed)  # Default to zero speed

    # Normalize speed for color mapping
    min_speed, max_speed = np.nanmin(filtered_speed), np.nanmax(filtered_speed)

    print("instanciating folium map")

    # Create map centered at the first coordinate
    m = folium.Map(location=[filtered_lat[0], filtered_lon[0]], zoom_start=14)

    # Plot segments with color-coded speed
    for i in range(len(filtered_lat) - 1):
        color = f'#{speed_to_color(filtered_speed[i], min_speed, max_speed)[0]:02x}' \
                f'{speed_to_color(filtered_speed[i], min_speed, max_speed)[1]:02x}' \
                f'{speed_to_color(filtered_speed[i], min_speed, max_speed)[2]:02x}'
        
        folium.PolyLine(
            [(filtered_lat[i], filtered_lon[i]), (filtered_lat[i + 1], filtered_lon[i + 1])],
            color=color,
            weight=5,
            opacity=0.8
        ).add_to(m)

    html_path = 'temp_map.html'
    m.save(html_path)
    begin_html_iframe = '<div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe srcdoc="'
    end_html_iframe = '" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none;"></iframe></div>'
                   
    with open(html_path, 'r') as file:
        html_as_string = file.read()
        map_html = gr.HTML(begin_html_iframe + html.escape(html_as_string) + end_html_iframe, visible=True)

    return map_html

# Create a Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("# 🌊 Surf Tracking - GPS Map")
    gr.Markdown("Upload a **.fit file** to visualize the GPS track on an interactive map.")

    file_input = gr.File(label="Upload .fit file")
    button = gr.Button("Analyze")
    output_map = gr.HTML()

    # Connect button to function
    button.click(plot_colored_route, inputs=file_input, outputs=output_map)

if __name__ == "__main__":
    demo.launch()

