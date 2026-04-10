'''
Copyright (c) 2023 Sylvain Combettes, Paul Boniol

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU Affero General Public License as published by the Free 
Software Foundation, either version 3 of the License, or (at your option) any 
later version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more 
details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np

from utils import *


def main(subject: int, movement: str, sensor: int):
    """
    Function to generate the Plotly animation with subplots.
    """

    # Set some configuration parameters
    path_line_sensors = [
        0,
        3,
        2,
        1,
        0,
        1,
        4,
        8,
        23,
        22,
        25,
        24,
        16,
        18,
        21,
        20,
        19,
        18,
        19,
        16,
        25,
        24,
        23,
        22,
        4,
        5,
        9,
        11,
        15,
        14,
        13,
        12,
        15,
        12,
        11,
        7,
        5,
        9,
        8,
        9,
        10,
        6,
        5,
        6,
        2,
        6,
        31,
        30,
        33,
        32,
        17,
        27,
        26,
        29,
        28,
        27,
        26,
        17,
        33,
        32,
        31,
        30,
        10,
    ]  # path line between the sensors
    sampling_period = 5  # time between two consecutive frames (a frame being a
    # downsample for the animation)
    frame_duration = 10  # frame duration (in the animation)
    list_of_colors = [
        "orange",
        "green",
        "violet",
    ]  # for each iteration of a movement

    # Load the data: multivariate time series and meta data
    ts, metadata = read_data(subject=subject, movement=movement)
    #ts, metadata = read_data_offline(subject=subject, movement=movement)

    # Print the metadata
    print_metadata(metadata=metadata)

    # Get some useful variables
    n_samples = ts.shape[1]  # number of samples
    n_frames = n_samples // sampling_period  # number of frames in the animation
    n_sensors = ts.shape[0]  # number of sensors (e.g. dimensions)
    sensor_names = [f"sensor #{i}" for i in range(n_sensors)]
    x, y, z = ts[:, :, 0].T, ts[:, :, 1].T, ts[:, :, 2].T  # spatial coordinates

    # Define the figure and generate the subplots
    fig = make_subplots(
        rows=4,
        cols=2,
        row_heights=[0.7, 0.1, 0.1, 0.1],
        column_widths=[0.5, 0.5],
        specs=[
            [{"rowspan": 4, "type": "scene"}, {"type": "scene"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D-trajectories of all sensors",
            f"3D-trajectory of sensor #{sensor}",
            f"Time series x of sensor #{sensor}",
            f"Time series y of sensor #{sensor}",
            f"Time series z of sensor #{sensor}",
        ),
    )

    # Intialize the frames: plot the lines and markers
    fig.add_trace(
        go.Scatter3d(
            x=x[0],
            y=y[0],
            z=z[0],
            mode="markers",
            marker=dict(color="blue", size=2),
            hovertext=sensor_names,
            hoverinfo="text",
            name="sensors",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x[0][sensor]],
            y=[y[0][sensor]],
            z=[z[0][sensor]],
            mode="markers",
            marker=dict(color="red", size=5),
            hovertext=f"selected sensor #{sensor}",
            hoverinfo="text",
            name=f"selected sensor #{sensor}",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=x[0][path_line_sensors],
            y=y[0][path_line_sensors],
            z=z[0][path_line_sensors],
            mode="lines",
            line=dict(color="grey", width=1),
            name="graph of sensors",
            hoverinfo="none",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        generate_scatter3d_trajectory(
            ts=ts,
            sensor=sensor,
            color="blue",
            name=f"3D-trajectory of sensor #{sensor}",
            interval=None,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(generate_ts_plot(ts, sensor, "blue", dim=0), row=2, col=2)
    fig.add_trace(generate_ts_plot(ts, sensor, "blue", dim=1), row=3, col=2)
    fig.add_trace(generate_ts_plot(ts, sensor, "blue", dim=2), row=4, col=2)
    # Initiliaze the frames: plot the time markers
    fig.add_trace(generate_scatter3d_marker(ts, sensor, "red"), row=1, col=2)
    fig.add_trace(unitialize_ts_plot_marker(ts, sensor, "red", dim=0), row=2, col=2)
    fig.add_trace(unitialize_ts_plot_marker(ts, sensor, "red", dim=1), row=3, col=2)
    fig.add_trace(unitialize_ts_plot_marker(ts, sensor, "red", dim=2), row=4, col=2)

    # Add the annotation meta data of the iterations
    for mov, color_mov in zip(
        metadata["Movement_label"].keys(), list_of_colors
    ):
        interval = metadata["Movement_label"][mov]
        mov_name_split = mov.split("_")
        mov_name = mov_name_split[0].lower() + " #" + mov_name_split[1]
        start_pos = int(interval[0])
        if np.isnan(interval[1]):
            end_pos = len(ts[sensor])
        else:
            end_pos = int(interval[1])
        fig.add_trace(
            generate_scatter3d_trajectory(
                ts,
                sensor,
                color=color_mov,
                name=mov_name,
                interval=[start_pos, end_pos],
            ),
            row=1,
            col=2,
        )
        add_annotation(
            fig, ts, sensor, 0, start_pos, end_pos, color_mov, mov_name
        )
        add_annotation(
            fig, ts, sensor, 1, start_pos, end_pos, color_mov, mov_name
        )
        add_annotation(
            fig, ts, sensor, 2, start_pos, end_pos, color_mov, mov_name
        )

    # Update the layout
    buttons = get_buttons(frame_duration=frame_duration)
    b_axis_range = get_axis_range(x, y, z)
    fig.update_yaxes(title_text="x", row=2, col=2)
    fig.update_yaxes(title_text="y", row=3, col=2)
    fig.update_yaxes(title_text="z", row=4, col=2)
    fig.update_xaxes(title_text="time stamp", row=4, col=2)
    fig.update_layout(
        sliders=get_sliders(
            n_frames=n_frames,
            frame_duration=frame_duration,
            sampling_period=sampling_period,
        ),
        updatemenus=[
            dict(
                type="buttons",
                x=0.03,
                y=0.0,
                xanchor="right",
                yanchor="top",
                pad=dict(r=70, t=40),
                buttons=buttons,
            )
        ],
        template="simple_white",
        hovermode="x unified",
        title_text=(
            f"Exploring the arm-CODA data set with a focus on movement {movement}"
            f" of subject #{subject} and sensor #{sensor}"
        ),
        scene=dict(
            xaxis=b_axis_range.xaxis,
            yaxis=b_axis_range.yaxis,
            zaxis=b_axis_range.zaxis,
            aspectratio=dict(x=1, y=1, z=b_axis_range.dz/b_axis_range.dxy),
        ),
    )

    # Update the frames
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=x[sampling_period * (k + 1)],
                    y=y[sampling_period * (k + 1)],
                    z=z[sampling_period * (k + 1)],
                    mode="markers",
                ),
                go.Scatter3d(
                    x=[x[sampling_period * (k + 1)][sensor]],
                    y=[y[sampling_period * (k + 1)][sensor]],
                    z=[z[sampling_period * (k + 1)][sensor]],
                    mode="markers",
                ),
                go.Scatter3d(
                    x=x[sampling_period * (k + 1)][path_line_sensors],
                    y=y[sampling_period * (k + 1)][path_line_sensors],
                    z=z[sampling_period * (k + 1)][path_line_sensors],
                    hoverinfo="none",
                    mode="lines",
                    name="graph of sensors",
                    line=dict(color="grey", width=1),
                ),
                go.Scatter3d(visible=True),
                go.Scatter(visible=True),
                go.Scatter(visible=True),
                go.Scatter(visible=True),
                go.Scatter3d(
                    x=[x[sampling_period * (k + 1)][sensor]],
                    y=[y[sampling_period * (k + 1)][sensor]],
                    z=[z[sampling_period * (k + 1)][sensor]],
                ),
                go.Scatter(
                    x=[sampling_period * (k + 1)],
                    y=[x[sampling_period * (k + 1)][sensor]],
                ),
                go.Scatter(
                    x=[sampling_period * (k + 1)],
                    y=[y[sampling_period * (k + 1)][sensor]],
                ),
                go.Scatter(
                    x=[sampling_period * (k + 1)],
                    y=[z[sampling_period * (k + 1)][sensor]],
                ),
            ],
            traces=list(np.arange(11)),
            name="{:.2f} s.".format(k / 100 * sampling_period),
        )
        for k in range(n_frames)
    ]
    fig.frames = frames
    fig.update(frames=frames)

    # Export into HTML format
    fig.write_html(
        "output.html",
        include_plotlyjs="cdn",
        default_width=1100,
        default_height=800,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--movement", type=str, required=True)
    parser.add_argument("--sensor", type=int, required=True)

    args = parser.parse_args()

    main(args.subject, args.movement, args.sensor)

    """
    Example of execution from the command line
    $ python3 main.py --subject 4 --movement "AEBPSA" --sensor 0
    """
