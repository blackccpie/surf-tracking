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

from wave_analyzer import wave_analyzer

# Constants for color gradient
COLORS = [
    (13, 8, 135),  # Dark blue
    (156, 23, 158),  # Purple
    (237, 121, 83),  # Orange
    (240, 249, 33)  # Yellow
]

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

    return interpolate_color(norm_speed, COLORS)

def create_map(filtered_lat, filtered_lon, filtered_speed, min_speed, max_speed, waves):
    """
    Creates a folium map with color-coded segments and wave markers.
    """
    
    # Calculate bounds for the map
    bounds = [[min(filtered_lat), min(filtered_lon)], [max(filtered_lat), max(filtered_lon)]]
    
    m = folium.Map(location=[filtered_lat[0], filtered_lon[0]], zoom_start=15, max_zoom=20)

    raw_positioning = folium.FeatureGroup(name="Raw Positioning", show=True)
    for i in range(len(filtered_lat) - 1):
        color = f'#{speed_to_color(filtered_speed[i], min_speed, max_speed)[0]:02x}' \
                f'{speed_to_color(filtered_speed[i], min_speed, max_speed)[1]:02x}' \
                f'{speed_to_color(filtered_speed[i], min_speed, max_speed)[2]:02x}'
        
        folium.PolyLine(
            [(filtered_lat[i], filtered_lon[i]), (filtered_lat[i + 1], filtered_lon[i + 1])],
            color=color,
            weight=5,
            opacity=0.8
        ).add_to(raw_positioning)

    wave_markers = folium.FeatureGroup(name="Wave Markers", show=False)
    wave_polylines = folium.FeatureGroup(name="Wave Polylines", show=True)

    for idx, wave_dict in enumerate(waves, start=1):
        wave_max_speed_idx = wave_dict['max_speed_index']
        wave_max_speed = wave_dict['max_speed']
        wave_first_idx = wave_dict['first_point_index']
        wave_last_idx = wave_dict['last_point_index']

        popup_text = f"#{idx}\n{round(3.6 * wave_max_speed,1)} km/h"
        folium.Marker(
            location=[filtered_lat[wave_max_speed_idx], filtered_lon[wave_max_speed_idx]],
            popup=popup_text,
            icon=folium.Icon("green")
        ).add_to(wave_markers)

        wave_points = [(filtered_lat[i], filtered_lon[i]) for i in range(wave_first_idx, wave_last_idx + 1)]
        
        folium.Circle(
            location=wave_points[0],
            radius=2,
            color="black",
            weight=1,
            fill_opacity=0.6,
            opacity=1,
            fill_color="green",
            fill=False,
        ).add_to(wave_polylines)

        folium.Circle(
            location=wave_points[-1],
            radius=2,
            color="black",
            weight=1,
            fill_opacity=0.6,
            opacity=1,
            fill_color="red",
            fill=False,
        ).add_to(wave_polylines)

        folium.PolyLine(
            wave_points,
            color="orange",
            weight=5,
            opacity=0.8
        ).add_to(wave_polylines)

    m.add_child(raw_positioning)
    m.add_child(wave_markers)
    m.add_child(wave_polylines)
    folium.LayerControl().add_to(m)

    # Fit the map to the bounds of the wave points
    m.fit_bounds(bounds)

    return m

def plot_colored_route(wazer: wave_analyzer):
    """
    Plots an activity map with segments color-coded by speed.
    Additionally, detects waves when speed exceeds 'wave_speed_threshold' (m/s) for at least 'wave_min_duration' seconds,
    and adds numbered markers to the map at the detected wave positions.
    """

    progress(0, desc="Initializing")

    # Retrieve wave data
    filtered_lat, filtered_lon, filtered_speed, min_speed, max_speed = wazer.get_motion_data()
    waves = wazer.get_waves_data()

    progress(0.5, desc="Processing wave data")

    print("instanciating folium map")

    m = create_map(filtered_lat, filtered_lon, filtered_speed, min_speed, max_speed, waves)

    progress(0.9, desc="Adding wave markers")

    html_path = 'temp_map.html'
    m.save(html_path)
    begin_html_iframe = '<div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe srcdoc="'
    end_html_iframe = '" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none;"></iframe></div>'
                   
    with open(html_path, 'r') as file:
        html_as_string = file.read()
        map_html = gr.HTML(begin_html_iframe + html.escape(html_as_string) + end_html_iframe, visible=True)

    progress(1, desc="Completed")

    return map_html

def analyze_waves(fit_file_path: str, wave_params: gr.State):
    """
    Analyzes waves from a .fit file and returns a map and a markdown table.
    """
    progress(0, desc="Initializing analysis")

    # Extract wave detection parameters from the state
    wave_speed_threshold = wave_params.get("speed_threshold", 2.0)
    wave_min_duration = wave_params.get("min_duration", 2.0)

    print(f"analyzing {fit_file_path}")

    # Analyze waves
    progress(0.1, desc="Initializing wave analyzer")
    wazer = wave_analyzer(wave_speed_threshold, wave_min_duration)

    progress(0.2, desc="Processing .fit file")
    wazer.process(fit_file_path)

    progress(0.5, desc="Generating map")
    map = plot_colored_route(wazer)

    progress(0.7, desc="Generating waves table")
    table = wazer.generate_waves_markdown_table()

    progress(0.9, desc="Generating session summary")
    summary = wazer.generate_summary_markdown_table()

    progress(1, desc="Analysis complete")
    return map, table, summary

# Create a Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("# 🌊 Surf Tracking - GPS Map")
    gr.Markdown("Upload a **.fit file** to visualize the GPS track on an interactive map.")

    with gr.Sidebar():
        file_input = gr.File(label="Upload .fit file", file_count='single', file_types=['.fit'], height=150)
        button = gr.Button("Analyze")
        
        with gr.Accordion("Detection Parameters", open=False):
            # Define sliders for wave detection parameters
            wave_threshold_slider = gr.Slider(label="Wave Detection Speed Threshold", minimum=0, maximum=5, value=2.0, step=0.1, interactive=True)
            wave_duration_slider = gr.Slider(label="Wave Detection Minimum Wave Duration", minimum=0, maximum=10, value=2.0, step=0.5, interactive=True)
        
    # Initialize state with default parameters
    wave_params = gr.State({"speed_threshold": 2.0, "min_duration": 2.0})

    gr.Markdown("## Session summary")
    waves_summary = gr.Markdown()
    gr.Markdown("## Session Map")
    output_map = gr.HTML()
    gr.Markdown("## Session details")
    waves_table = gr.Markdown()
    progress = gr.Progress()

    def update_wave_params(speed_threshold, min_duration):
        return {"speed_threshold": speed_threshold, "min_duration": min_duration}

    # Update the state when either slider changes
    wave_threshold_slider.change(update_wave_params, 
                                inputs=[wave_threshold_slider, wave_duration_slider], 
                                outputs=wave_params)
    wave_duration_slider.change(update_wave_params, 
                                inputs=[wave_threshold_slider, wave_duration_slider], 
                                outputs=wave_params)

    # Connect button to function
    button.click(analyze_waves, inputs=[file_input, wave_params], outputs=[output_map, waves_table, waves_summary])

if __name__ == "__main__":
    demo.launch()

