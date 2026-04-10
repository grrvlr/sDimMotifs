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


import json
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.utils import Bunch

"""
Dockerfile
RUN wget "https://kiwi.cmla.ens-cachan.fr/index.php/s/yLT6TiyiwXGB54t/download?path=/77777000460/Data&files=" -O my_folder.zip
maintenant mon fichier est dans le bin sur IPOL et il faut que je le unzip et ensuite j'ai mon folder comme il faut
"""

def read_data(subject: int, movement: str):
    """
    Read the Arm-CODA data set that is stored on the Nextcloud.
    """

    address_ts = "https://kiwi.cmla.ens-cachan.fr/index.php/s/yLT6TiyiwXGB54t/download?path=/77777000460/dataset&files="
    address_metadata = "https://kiwi.cmla.ens-cachan.fr/index.php/s/yLT6TiyiwXGB54t/download?path=/77777000460/dataset&files="

    filename_ts = (
        "armcoda_subject" + "{}".format(subject) + "_movement" + movement + ".npy"
    )
    filename_meta = (
        "armcoda_subject" + "{}".format(subject) + "_movement" + movement + ".json"
    )

    urllib.request.urlretrieve(address_ts + filename_ts, filename_ts)
    urllib.request.urlretrieve(address_metadata + filename_meta, filename_meta)

    ts = np.load(filename_ts, allow_pickle=True)
    with open(filename_meta, "r") as f:
        metadata = json.load(f)

    return ts, metadata


def read_data_offline(subject, movement):
    """
    Read the data when you have no internet connection.
    It requires to have the data already downloaded on your laptop.
    """
    if subject < 10:
        str_subject = "0" + str(subject)
    else:
        str_subject = str(subject)
    ts = np.load(f"../Data_Smartarm_MC/smartarm_gapfree/armcoda_subject{str_subject}_movement{movement}.npy")
    filename_meta = f"../Data_Smartarm_MC/MetaData/armcoda_subject{str_subject}_movement{movement}.json"
    with open(filename_meta, 'r') as f:
        metadata = json.load(f)
    return ts, metadata


def print_metadata(metadata):
    """`metadata` is loaded from a json file"""

    print("Subject info:\n---")
    df_subject = pd.DataFrame.from_dict(
        metadata["Patient_info"], orient="index"
    ).T
    print(df_subject.to_string(index=False, justify="right"))

    print("\nMovement info:\n---")
    df_movement_info = pd.DataFrame.from_dict(
        metadata["Movement_info"], orient="index"
    )
    print(df_movement_info.to_string(header=None, justify="left"))

    print("\nMovement annotation:\n---")
    df_movement_annotation = pd.DataFrame.from_dict(
        metadata["Movement_label"], orient="index"
    ).rename(columns={0: "start", 1: "end"})
    print(df_movement_annotation.to_string(justify="right"))


def generate_scatter3d_trajectory(ts, sensor, color, name, interval=None):
    """
    Generate a full trajectory in 3D.
    """
    if interval is None:
        interval = [0, len(ts[sensor])]
    return go.Scatter3d(
        x=ts[sensor][interval[0]:interval[1], 0],
        y=ts[sensor][interval[0]:interval[1], 1],
        z=ts[sensor][interval[0]:interval[1], 2],
        mode="lines",
        name=name,
        line={"color": color},
        showlegend=True,
    )


def generate_ts_plot(ts, sensor, color, dim):
    """
    Generate a univariate time series plot amplitude vs time.
    """
    ts_sensor_dim = ts[sensor][:, dim]
    return go.Scatter(
        x=list(range(len(ts_sensor_dim))),
        y=ts_sensor_dim,
        line={"color": color},
        showlegend=False,
    )


def generate_scatter3d_marker(ts, sensor, color):
    return go.Scatter3d(
        x=[ts[sensor][0, 0]],
        y=[ts[sensor][0, 1]],
        z=[ts[sensor][0, 2]],
        mode="markers",
        marker={"color": color, "size": 7},
        showlegend=False,
    )


def unitialize_ts_plot_marker(ts, sensor, color, dim):
    return go.Scatter(
        x=[0],
        y=[ts[sensor][0, dim]],
        mode="markers",
        marker={"color": color, "size": 10},
        showlegend=False,
    )


def add_annotation(fig, ts, sensor, dim, start_pos, end_pos, color, name):
    ts_min = min(ts[sensor][:, dim])
    ts_max = max(ts[sensor][:, dim])
    coeff = 0
    fig.add_shape(
        type="rect",
        row=2 + dim,
        col=2,
        x0=start_pos,
        x1=end_pos,
        y0=ts_min - coeff * ts_min,
        y1=ts_max + coeff * ts_max,
        fillcolor=color,
        opacity=0.2,
        layer="below",
        line_width=0.2,
    )
    fig.add_annotation(
        x=(start_pos + end_pos) // 2,
        y=min(ts[sensor][:, dim]),
        row=2 + dim,
        col=2,
        text=name,
        showarrow=False,
        # yshift=(max(ts[sensor][:,dim])-min(ts[sensor][:,dim]))//10
    )
    return fig


def get_sliders(
    n_frames, sampling_period, frame_duration=100, x_pos=0.0, slider_len=1.0
):
    """
    Get the sliders for the Plotly animation.

    Parameters
    ----------
    n_frames:
        number of frames

    frame_duration:
        the duration in milliseconds of each frame

    x_pos:
        x-coordinate where the slider starts

    slider_len:
        number in (0,1] giving the slider length as a fraction of x-axis length
    """
    return [
        dict(
            steps=[
                dict(
                    method="animate",  # Sets the Plotly method to be called when the slider value is changed.
                    args=[
                        [
                            "{:.2f} s.".format(k / 100 * sampling_period)
                        ],  # Sets the arguments values to be passed to the Plotly,
                        # method set in method on slide
                        dict(
                            mode="immediate",
                            frame=dict(duration=frame_duration, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label="{:.2f} s.".format(k / 100 * sampling_period),
                )
                for k in range(n_frames)
            ],
            transition={"duration": 0},
            x=x_pos,
            len=slider_len,
        )
    ]

def get_buttons(frame_duration):
    """
    Get the Play and Pause buttons for Plotly animations.
    """
    buttons = [
        dict(
            label="Play",
            method="animate",
            args=[
                None,
                dict(
                    mode="immediate",
                    transition={"duration": 0},
                    fromcurrent=True,
                    frame=dict(redraw=True, duration=frame_duration),
                ),
            ],
        ),
        dict(
            label="Pause",
            method="animate",
            args=[
                [None],
                dict(
                    mode="immediate",
                    transition={"duration": 0},
                    frame=dict(redraw=True, duration=0),
                ),
            ],
        ),
    ]
    return buttons

def get_axis_range(x, y, z):
    
    xm, xM = x.min(), x.max()
    ym, yM = y.min(), y.max()
    zm, zM = z.min(), z.max()

    xmid, ymid, zmid = (xm + xM) / 2, (ym + yM) / 2, (zm + zM) / 2
    dx, dy, dz = (xM - xm) / 2, (yM - ym) / 2, (zM - zm) / 2
    dxy = max(dx, dy) * 1.05
    dz = dz * 1.05

    xaxis = dict(range=[xmid - dxy, xmid + dxy], autorange=False)
    yaxis = dict(range=[ymid - dxy, xmid + dxy], autorange=False)
    zaxis = dict(range=[zmid - dz, zmid + dz], autorange=False)

    b_axis_range = Bunch(
        dxy=dxy,
        dz=dz,
        xaxis=xaxis,
        yaxis=yaxis,
        zaxis=zaxis,
    )
    return b_axis_range
