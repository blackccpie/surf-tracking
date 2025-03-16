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

import numpy as np

from fitparse import FitFile
from pykalman import KalmanFilter
from geopy.distance import geodesic

class wave_analyzer:
    """
    Manages all wave analysis process from gps data.
    """

    def __init__(self, speed_threshold, min_duration):
        self.latitudes = np.array([])
        self.longitudes = np.array([])
        self.filtered_latitudes = []
        self.filtered_longitudes = []
        self.filtered_speeds = np.array([])
        self.timestamps = np.array([])
        self.speeds = []
        self.waves = []
        self.min_speed = 0.0
        self.max_speed = 0.0

        self.speed_threshold = speed_threshold
        self.min_duration = min_duration

        print(f"speed threshold set to: {self.speed_threshold}")
        print(f"min duration set to: {self.min_duration}")

    def __extract_fit_data(self, fit_file_path):
        """Extracts lat, lon, and speed (if available) from a .fit file."""
        fitfile = FitFile(fit_file_path)

        _latitudes, _longitudes, _speeds, _timestamps = [], [], [], []

        _num_records = 0

        for _record in fitfile.get_messages("record"):
            lat, lon, speed, timestamp = self.__parse_record(_record)

            if lat is not None and lon is not None:
                _latitudes.append(lat)
                _longitudes.append(lon)
                _timestamps.append(timestamp if timestamp is not None else 0)
                _speeds.append(speed if speed is not None else np.nan)  # Use NaN for missing speeds

            _num_records += 1

        if not _latitudes:
            print("No GPS data found in the file!")

        print(f"Finished parsing {_num_records} records")

        self.latitudes = np.array(_latitudes)
        self.longitudes = np.array(_longitudes)
        self.speeds = np.array(_speeds)
        self.timestamps = np.array(_timestamps)

    def __parse_record(self, record):
        """Parses a single record from the .fit file."""
        lat, lon, speed, timestamp = None, None, None, None

        for _data in record:
            if _data.name == "position_lat":
                lat = _data.value / (2**31) * 180  # Convert Garmin's semi-circle format to degrees
            elif _data.name == "position_long":
                lon = _data.value / (2**31) * 180
            elif _data.name == "enhanced_speed":  # Speed in m/s
                speed = _data.value
            elif _data.name == "timestamp":
                timestamp = _data.value.timestamp()  # Convert to UNIX time

        return lat, lon, speed, timestamp

    def __compute_speed_from_gps(self):
        """
        Estimates speed (m/s) from GPS data using Haversine distance and time.
        """
        
        _estimated_speeds = np.zeros_like(self.latitudes, dtype=float)

        for i in range(1, len(self.latitudes)):
            distance = geodesic((self.latitudes[i-1], self.longitudes[i-1]), (self.latitudes[i], self.longitudes[i])).meters
            time_diff = self.timestamps[i] - self.timestamps[i-1]

            _estimated_speeds[i] = distance / time_diff if time_diff > 0 else 0  # Avoid division by zero

        return _estimated_speeds

    def __apply_kalman_filter(
            self, 
            transition_covariance_ratio=0.00001, # Adjust for smoother tracking
            observation_covariance_ratio=0.0001 # Adjust GPS noise level
        ):
        """
        Applies a Kalman Filter to smooth lat, lon, and estimated speed.
        """
        if np.isnan(self.speeds).all():  # If all speed values are missing, estimate them
            self.speeds = self.__compute_speed_from_gps()
        else:
            print("using native speed measurements")

        initial_state = [self.latitudes[0], self.longitudes[0], self.speeds[0]]

        transition_matrix = np.eye(3)  # Identity matrix (assumes smooth movement)
        observation_matrix = np.eye(3)  # Direct observation

        kf = KalmanFilter(
            initial_state_mean=initial_state,
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=np.eye(3) * transition_covariance_ratio,
            observation_covariance=np.eye(3) * observation_covariance_ratio,
        )

        smoothed_states, _ = kf.smooth(np.column_stack([self.latitudes, self.longitudes, self.speeds]))
        return smoothed_states[:, 0], smoothed_states[:, 1], smoothed_states[:, 2]  # Smoothed lat, lon, speed
    
    def __detect_waves(self, input_speeds):
        """
        Detects waves when the speed stays above 'speed_threshold' for at least 'min_duration' seconds.
        Returns a list of metadata corresponding to detected wave.
        """
        waves = []
        in_wave = False
        start_idx = None

        def __add_wave(duration, max_speed_index, wave_segment_indices, start_idx, end_idx):
            wave_length = sum(
                geodesic((self.latitudes[i], self.longitudes[i]), (self.latitudes[i + 1], self.longitudes[i + 1])).meters
                for i in wave_segment_indices[:-1]
            )
            waves.append({
                "max_speed_index": max_speed_index,
                "max_speed": input_speeds[max_speed_index],
                "duration": duration,
                "num_points": len(wave_segment_indices),
                "first_point_index": start_idx,
                "last_point_index": end_idx,
                "length": wave_length
            })

        for i, speed in enumerate(input_speeds):
            if speed >= self.speed_threshold:
                if not in_wave:
                    in_wave = True
                    start_idx = i
            else:
                if in_wave:
                    # Wave segment ended: check if duration qualifies
                    duration = self.timestamps[i - 1] - self.timestamps[start_idx]
                    if duration >= self.min_duration:
                        # Choose the point of maximum speed in the segment as the wave "event"
                        wave_segment_indices = range(start_idx, i)
                        max_speed_index = max(wave_segment_indices, key=lambda j: input_speeds[j])
                        __add_wave(duration, max_speed_index, wave_segment_indices, start_idx, i)
                    in_wave = False
        # In case the final segment is still in-wave:
        if in_wave:
            duration = self.timestamps[-1] - self.timestamps[start_idx]
            if duration >= self.min_duration:
                wave_segment_indices = range(start_idx, len(input_speeds))
                max_speed_index = max(wave_segment_indices, key=lambda j: input_speeds[j])
                __add_wave(duration, max_speed_index, wave_segment_indices, start_idx, i)
        return waves

    def __filter_outlier_waves(self, 
                               waves, 
                               max_speed_threshold_kmh=15.0, 
                               min_duration_threshold=1.0, 
                               max_duration_threshold=15.0):
        """
        Filters out waves that are considered outliers based on speed (km/h) and duration thresholds (s).
        """
        filtered_waves = []
        for wave in waves:
            if wave['max_speed'] <= (max_speed_threshold_kmh/3.6) and wave['duration'] >= min_duration_threshold and wave['duration'] <= max_duration_threshold:
                filtered_waves.append(wave)
        return filtered_waves
    
    def get_motion_data(self):
        """
        Retrieves computed motion data.
        """
        return self.filtered_latitudes, self.filtered_longitudes, self.filtered_speeds, self.min_speed, self.max_speed

    def get_waves_data(self):
        """
        Retrieves computed waves data.
        """
        return self.waves

    def process(self, fit_file_path, disable_filter=False, transition_covariance_ratio=0.00001, observation_covariance_ratio=0.0001):
        """
        Processes waves data file.
        """

        # Parse .fit data
        self.__extract_fit_data(fit_file_path)

        if not disable_filter:
            # Apply Kalman filter (with speed estimation if missing)
            self.filtered_latitudes, self.filtered_longitudes, self.filtered_speeds = self.__apply_kalman_filter(
                transition_covariance_ratio=transition_covariance_ratio,
                observation_covariance_ratio=observation_covariance_ratio
            )
        else:
            self.filtered_latitudes, self.filtered_longitudes, self.filtered_speeds = self.latitudes, self.longitudes, self.speeds

        # Handle missing or constant speed values
        if np.isnan(self.filtered_speeds).all() or (np.max(self.filtered_speeds) == np.min(self.filtered_speeds)):
            print("Warning: No valid speed variation detected. Using default color.")
            self.filtered_speeds = np.zeros_like(self.filtered_speeds)  # Default to zero speed

        # Normalize speed for color mapping
        self.min_speed, self.max_speed = np.nanmin(self.filtered_speeds), np.nanmax(self.filtered_speeds)

        # Detect waves using the specified speed and duration thresholds
        self.waves = self.__detect_waves(self.filtered_speeds)

        # Filter outlier waves
        self.waves = self.__filter_outlier_waves(self.waves)

        print(f"detected {len(self.waves)} waves")

    def generate_waves_markdown_table(self):
        """
        Generates a markdown formatted table representing self.waves's data.
        """
        if not self.waves:
            return "No waves detected."

        table_header = "| Wave Index | Max Speed (km/h) | Duration (s) | Number of Points | Start Index | End Index | Length (m) |\n"
        table_divider = "|------------|------------------|--------------|------------------|-------------|-----------|------------|\n"
        table_rows = ""

        for idx, wave in enumerate(self.waves, start=1):
            max_speed_kmh = wave['max_speed'] * 3.6  # Convert m/s to km/h
            table_rows += f"| {idx} | {max_speed_kmh:.2f} | {wave['duration']} | {wave['num_points']} | {wave['first_point_index']} | {wave['last_point_index']} | {wave['length']:.2f} |\n"

        return table_header + table_divider + table_rows

