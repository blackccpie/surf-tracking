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

def plot_colored_route(fit_file_path, wave_params):
    """
    Plots an activity map with segments color-coded by speed.
    Additionally, detects waves when speed exceeds 'wave_speed_threshold' (m/s) for at least 'wave_min_duration' seconds,
    and adds numbered markers to the map at the detected wave positions.
    """

    progress(0, desc="Initializing")

    # Extract wave detection parameters from the state
    wave_speed_threshold = wave_params.get("speed_threshold", 2.0)
    wave_min_duration = wave_params.get("min_duration", 2.0)
    
    print(f"plotting {fit_file_path}")

    # Analyze wave data
    wazer = wave_analyzer( wave_speed_threshold, wave_min_duration )
    wazer.process(fit_file_path)
    filtered_lat, filtered_lon, filtered_speed, min_speed, max_speed = wazer.get_motion_data()
    waves = wazer.get_waves_data()

    progress(0.5, desc="Processing wave data")

    print("instanciating folium map")

    # Create map centered at the first coordinate
    m = folium.Map(location=[filtered_lat[0], filtered_lon[0]], zoom_start=15, max_zoom=20)

     # Create a FeatureGroup for raw positioning
    raw_positioning = folium.FeatureGroup(name="Raw Positioning")

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
        ).add_to(raw_positioning)

    progress(0.7, desc="Plotting segments")

    # Create a FeatureGroup for wave markers
    wave_markers = folium.FeatureGroup(name="Wave Markers")

    # Create a FeatureGroup for wave segments
    wave_segments = folium.FeatureGroup(name="Wave Segments")

    # Add markers with wave index numbers at detected wave locations
    for idx, wave_dict in enumerate(waves, start=1):
        wave_max_speed_idx = wave_dict['max_speed_index']
        wave_max_speed = wave_dict['max_speed']
        wave_first_idx = wave_dict['first_point_index']
        wave_last_idx = wave_dict['last_point_index']

        print(f"{wave_max_speed_idx}/{wave_first_idx}/{wave_last_idx}")

        popup_text = f"#{idx}\n{round(3.6 * wave_max_speed,1)} km/h"
        folium.Marker(
            location=[filtered_lat[wave_max_speed_idx], filtered_lon[wave_max_speed_idx]],
            popup=popup_text,
            #icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black">{idx}</div>')
            icon=folium.Icon("green")
        ).add_to(wave_markers)

        folium.Circle(
            location=[filtered_lat[wave_first_idx], filtered_lon[wave_first_idx]],
            radius=2,
            color="black",
            weight=1,
            fill_opacity=0.6,
            opacity=1,
            fill_color="green",
            fill=False,  # gets overridden by fill_color
        ).add_to(wave_segments)

        folium.Circle(
            location=[filtered_lat[wave_last_idx], filtered_lon[wave_last_idx]],
            radius=2,
            color="black",
            weight=1,
            fill_opacity=0.6,
            opacity=1,
            fill_color="red",
            fill=False,  # gets overridden by fill_color
        ).add_to(wave_segments)

        folium.PolyLine(
            [(filtered_lat[wave_first_idx], filtered_lon[wave_first_idx]), (filtered_lat[wave_last_idx], filtered_lon[wave_last_idx])],
            color="green",
            weight=5,
            opacity=0.8
        ).add_to(wave_segments)

    progress(0.9, desc="Adding wave markers")

    # Add the layers to the map
    m.add_child(raw_positioning)
    m.add_child(wave_markers)
    m.add_child(wave_segments)

    # Add a LayerControl so that layers can be toggled
    folium.LayerControl().add_to(m)

    html_path = 'temp_map.html'
    m.save(html_path)
    begin_html_iframe = '<div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe srcdoc="'
    end_html_iframe = '" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none;"></iframe></div>'
                   
    with open(html_path, 'r') as file:
        html_as_string = file.read()
        map_html = gr.HTML(begin_html_iframe + html.escape(html_as_string) + end_html_iframe, visible=True)

    progress(1, desc="Completed")

    return map_html

# Create a Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("# 🌊 Surf Tracking - GPS Map")
    gr.Markdown("Upload a **.fit file** to visualize the GPS track on an interactive map.")

    file_input = gr.File(label="Upload .fit file")
    
    with gr.Accordion("Detection Parameters", open=False):
        # Define sliders for wave detection parameters
        wave_threshold_slider = gr.Slider(label="Wave Detection Speed Threshold", minimum=0, maximum=5, value=2.0, step=0.1, interactive=True)
        wave_duration_slider = gr.Slider(label="Wave Detection Minimum Wave Duration", minimum=0, maximum=10, value=2.0, step=0.5, interactive=True)
    
    # Initialize state with default parameters
    wave_params = gr.State({"speed_threshold": 2.0, "min_duration": 2.0})

    button = gr.Button("Analyze")
    progress = gr.Progress()
    output_map = gr.HTML()

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
    button.click(plot_colored_route, inputs=[file_input, wave_params], outputs=output_map)

if __name__ == "__main__":
    demo.launch()

