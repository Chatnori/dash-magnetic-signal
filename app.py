### DASH
from dash import Dash, dcc, html, Output, Input
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import namedtuple

# 初始化 Dash 应用
app = Dash(__name__)

# 定义磁场参数
MagParam= namedtuple('MagParam', ['B', 'alpha', 'beta', 'gamma', 'f', 'epsilon','theta_sweep'])

# 设置初始参数
t = np.linspace(0, 1, 500)


def compute_signals(mag_mode_1, mag_mode_2, B1, B2, f1, f2, alpha1, beta1, gamma1, epsilon1, theta1, alpha2, beta2, gamma2, epsilon2, theta2):
    alpha1, alpha2, beta1, beta2, gamma1, gamma2, theta1, theta2 = np.radians([alpha1, alpha2, beta1, beta2, gamma1, gamma2, theta1, theta2])
    if mag_mode_1 == 'Ellipse':
        # 椭圆磁场
        # 计算第一个信号的相位角 phi_x1, phi_y1, phi_z1
        phi_x1 = np.arctan2(-np.cos(gamma1) * np.sin(alpha1) + np.cos(alpha1) * np.sin(beta1) * np.sin(gamma1),
                            epsilon1 * (np.cos(alpha1) * np.cos(gamma1) * np.sin(beta1) + np.sin(alpha1) * np.sin(gamma1)))

        phi_y1 = np.arctan2(np.cos(alpha1) * np.cos(gamma1) + np.sin(alpha1) * np.sin(beta1) * np.sin(gamma1),
                            epsilon1 * (np.cos(gamma1) * np.sin(alpha1) * np.sin(beta1) - np.cos(alpha1) * np.sin(gamma1)))

        phi_z1 = np.arctan2(np.cos(beta1) * np.sin(gamma1), epsilon1 * np.cos(beta1) * np.cos(gamma1))

        # 计算第一个信号的磁场分量 B_x1, B_y1, B_z1
        B_x1 = B1 * np.sqrt((-np.cos(gamma1) * np.sin(alpha1) + np.cos(alpha1) * np.sin(beta1) * np.sin(gamma1)) ** 2 +
                            (epsilon1 * (np.cos(alpha1) * np.cos(gamma1) * np.sin(beta1) + np.sin(alpha1) * np.sin(
                                gamma1))) ** 2) \
               * np.sin(2 * np.pi * f1 * t + phi_x1)

        B_y1 = B1 * np.sqrt((np.cos(alpha1) * np.cos(gamma1) + np.sin(alpha1) * np.sin(beta1) * np.sin(gamma1)) ** 2 +
                            (epsilon1 * (np.cos(gamma1) * np.sin(alpha1) * np.sin(beta1) - np.cos(alpha1) * np.sin(
                                gamma1))) ** 2) \
               * np.sin(2 * np.pi * f1 * t + phi_y1)

        B_z1 = B1 * np.sqrt((np.cos(beta1) * np.sin(gamma1)) ** 2 +
                            (epsilon1 * np.cos(beta1) * np.cos(gamma1)) ** 2) \
               * np.sin(2 * np.pi * f1 * t + phi_z1)
    elif mag_mode_1 == 'Sweep':
        # 振荡
        Tri_wave = 4 * theta1 * np.abs(t * f1 - np.floor(t * f1 + 0.75) + 0.25) - theta1

        phi_x1 = np.arctan2(np.cos(alpha1) * np.cos(beta1),
                           -np.cos(gamma1) * np.sin(alpha1) + np.cos(alpha1) * np.sin(beta1) * np.sin(gamma1))
        phi_y1 = np.arctan2(np.cos(beta1) * np.sin(alpha1),
                           np.cos(alpha1) * np.cos(gamma1) + np.sin(alpha1) * np.sin(beta1) * np.sin(gamma1))
        phi_z1 = np.arctan2(-np.sin(beta1), np.cos(beta1) * np.sin(gamma1))

        B_x1 = B1 * np.sqrt((np.cos(alpha1) * np.cos(beta1)) ** 2 + (
                    -np.cos(gamma1) * np.sin(alpha1) + np.cos(alpha1) * np.sin(beta1) * np.sin(gamma1)) ** 2) * np.sin(
            Tri_wave + phi_x1)
        B_y1 = B1 * np.sqrt((np.cos(beta1) * np.sin(alpha1)) ** 2 + (
                    np.cos(alpha1) * np.cos(gamma1) + np.sin(alpha1) * np.sin(beta1) * np.sin(gamma1)) ** 2) * np.sin(
            Tri_wave + phi_y1)
        B_z1 = B1 * np.sqrt(np.sin(beta1) ** 2 + (np.cos(beta1) * np.sin(gamma1)) ** 2) * np.sin(
            Tri_wave + phi_z1)
    elif mag_mode_1 == 'Static':
        B_x1 = B1 * np.cos(alpha1) * np.cos(beta1) * np.ones_like(t)
        B_y1 = B1 * np.cos(beta1) * np.sin(alpha1) * np.ones_like(t)
        B_z1 = -B1 * np.sin(beta1) * np.ones_like(t)


    if mag_mode_2 == 'Ellipse':
        # 椭圆磁场
        # 计算第二个信号的相位角 phi_x2, phi_y2, phi_z2
        phi_x2 = np.arctan2(np.cos(alpha2) * np.sin(beta2) + np.sin(alpha2) * np.cos(beta2) * np.sin(gamma2),
                            epsilon2 * (np.cos(alpha2) * np.cos(beta2) - np.sin(alpha2) * np.sin(beta2) * np.sin(gamma2)))

        phi_y2 = np.arctan2(np.sin(alpha2) * np.sin(beta2) - np.cos(alpha2) * np.cos(beta2) * np.sin(gamma2),
                            epsilon2 * (np.sin(alpha2) * np.cos(beta2) + np.cos(alpha2) * np.sin(beta2) * np.sin(gamma2)))

        phi_z2 = np.arctan2(np.cos(beta2) * np.cos(gamma2), -epsilon2 * np.sin(beta2) * np.cos(gamma2))

        # 计算第二个信号的磁场分量 B_x2, B_y2, B_z2
        B_x2 = B2 * np.sqrt((np.cos(alpha2) * np.sin(beta2) + np.sin(alpha2) * np.cos(beta2) * np.sin(gamma2)) ** 2 +
                            (epsilon2 * (np.cos(alpha2) * np.cos(beta2) - np.sin(alpha2) * np.sin(beta2) * np.sin(
                                gamma2))) ** 2) \
               * np.sin(2 * np.pi * f2 * t + phi_x2)

        B_y2 = B2 * np.sqrt((np.sin(alpha2) * np.sin(beta2) - np.cos(alpha2) * np.cos(beta2) * np.sin(gamma2)) ** 2 +
                            (epsilon2 * (np.sin(alpha2) * np.cos(beta2) + np.cos(alpha2) * np.sin(beta2) * np.sin(
                                gamma2))) ** 2) \
               * np.sin(2 * np.pi * f2 * t + phi_y2)

        B_z2 = B2 * np.sqrt((np.cos(beta2) * np.cos(gamma2)) ** 2 +
                            (-epsilon2 * np.sin(beta2) * np.cos(gamma2)) ** 2) \
               * np.sin(2 * np.pi * f2 * t + phi_z2)
    elif mag_mode_2 == 'Sweep':
        # 振荡
        Tri_wave = 4 * theta2 * np.abs(t * f2 - np.floor(t * f2 + 0.75) + 0.25) - theta2

        phi_x2 = np.arctan2(-np.sin(alpha2) * np.cos(gamma2),
                            np.cos(alpha2) * np.sin(beta2) + np.sin(alpha2) * np.cos(beta2) * np.sin(gamma2))
        phi_y2 = np.arctan2(np.cos(alpha2) * np.cos(gamma2),
                            np.sin(alpha2) * np.sin(beta2) - np.cos(alpha2) * np.cos(beta2) * np.sin(gamma2))
        phi_z2 = np.arctan2(np.sin(gamma2), np.cos(beta2) * np.cos(gamma2))

        B_x2 = B2 * np.sqrt((-np.sin(alpha2) * np.cos(gamma2)) ** 2 + (
                    np.cos(alpha2) * np.sin(beta2) + np.sin(alpha2) * np.cos(beta2) * np.sin(gamma2)) ** 2) * np.sin(
            Tri_wave + phi_x2)
        B_y2 = B2 * np.sqrt((np.cos(alpha2) * np.cos(gamma2)) ** 2 + (
                    np.sin(alpha2) * np.sin(beta2) - np.cos(alpha2) * np.cos(beta2) * np.sin(gamma2)) ** 2) * np.sin(
            Tri_wave + phi_y2)
        B_z2 = B2 * np.sqrt(np.sin(gamma2) ** 2 + (np.cos(beta2) * np.cos(gamma2)) ** 2) * np.sin(
            Tri_wave + phi_z2)
    elif mag_mode_2 == 'Static':
        B_x2 = -B2 * np.sin(alpha2) * np.cos(gamma2) * np.ones_like(t)
        B_y2 = B2 * np.cos(alpha2) * np.cos(gamma2) * np.ones_like(t)
        B_z2 = B2 * np.sin(gamma2) * np.ones_like(t)

    # 叠加两个磁场信号
    return B_x1, B_y1, B_z1, B_x2, B_y2, B_z2

# Layout for user input controls and graphs
app.layout = html.Div([
    # Field 1 Controls
    html.Div([
        html.H4("Magnetic Field 1 Mode"),
        dcc.RadioItems(
            ["Ellipse", "Sweep", "Static"], "Ellipse", id="mag-mode-1"
        ),
        html.H4("Magnetic Field 1 Parameters"),

        html.Label("B1 (mT)", style={'margin-right': '10px'}),
        dcc.Input(id='B1-input', type='number', value=1, step=0.1),
        html.Div(dcc.Slider(id='B1', min=0, max=5, step=0.1, value=1, marks={i: str(i) for i in range(6)}), style={'width': '50%'}),

        html.Label("f1 (Hz)", style={'margin-right': '10px'}),
        dcc.Input(id='f1-input', type='number', value=1, step=1),
        html.Div(dcc.Slider(id='f1', min=0, max=100, step=1, value=1, marks={10*i: str(10*i) for i in range(11)}), style={'width': '50%'}),

        html.Label("α1 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='alpha1-input', type='number', value=0, step=10),
        html.Div(dcc.Slider(id='alpha1', min=0, max=360, step=10, value=0, marks={0: '0', 180: '180', 360: '360'}), style={'width': '50%'}),

        html.Label("β1 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='beta1-input', type='number', value=0, step=10),
        html.Div(dcc.Slider(id='beta1', min=0, max=360, step=10, value=0, marks={0: '0', 180: '180', 360: '360'}), style={'width': '50%'}),

        html.Label("γ1 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='gamma1-input', type='number', value=0, step=10),
        html.Div(dcc.Slider(id='gamma1', min=0, max=360, step=10, value=0, marks={0: '0', 180: '180', 360: '360'}), style={'width': '50%'}),

        html.Label("ε1", style={'margin-right': '10px'}),
        dcc.Input(id='epsilon1-input', type='number', value=1, step=0.1),
        html.Div(dcc.Slider(id='epsilon1', min=0, max=1, step=0.1, value=1, marks={0.1*i: str(np.around(0.1*i,1)) for i in range(11)}), style={'width': '50%'}),

        html.Label("θ1 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='theta1-input', type='number', value=45, step=5),
        html.Div(dcc.Slider(id='theta1', min=0, max=180, step=5, value=45, marks={0: '0', 180: '180'}), style={'width': '50%'}),
    ]),

    # Field 2 Controls
    html.Div([
        html.H4("Magnetic Field 2 Mode"),
        dcc.RadioItems(
            ["Ellipse", "Sweep", "Static"], "Ellipse", id="mag-mode-2"
        ),
        html.H4("Magnetic Field 2 Parameters"),

        html.Label("B2 (mT)", style={'margin-right': '10px'}),
        dcc.Input(id='B2-input', type='number', value=1, step=0.1),
        html.Div(dcc.Slider(id='B2', min=0, max=5, step=0.1, value=1, marks={i: str(i) for i in range(6)}), style={'width': '50%'}),

        html.Label("f2 (Hz)", style={'margin-right': '10px'}),
        dcc.Input(id='f2-input', type='number', value=1, step=1),
        html.Div(dcc.Slider(id='f2', min=0, max=100, step=1, value=1, marks={10*i: str(10*i) for i in range(11)}), style={'width': '50%'}),

        html.Label("α2 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='alpha2-input', type='number', value=0, step=10),
        html.Div(dcc.Slider(id='alpha2', min=0, max=360, step=10, value=0, marks={0: '0', 180: '180', 360: '360'}), style={'width': '50%'}),

        html.Label("β2 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='beta2-input', type='number', value=0, step=10),
        html.Div(dcc.Slider(id='beta2', min=0, max=360, step=10, value=0, marks={0: '0', 180: '180', 360: '360'}), style={'width': '50%'}),

        html.Label("γ2 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='gamma2-input', type='number', value=0, step=10),
        html.Div(dcc.Slider(id='gamma2', min=0, max=360, step=10, value=0, marks={0: '0', 180: '180', 360: '360'}), style={'width': '50%'}),

        html.Label("ε2", style={'margin-right': '10px'}),
        dcc.Input(id='epsilon2-input', type='number', value=1, step=0.1),
        html.Div(dcc.Slider(id='epsilon2', min=0, max=1, step=0.1, value=1, marks={0.1*i: str(np.around(0.1*i,1)) for i in range(11)}), style={'width': '50%'}),

        html.Label("θ2 (°)", style={'margin-right': '10px'}),
        dcc.Input(id='theta2-input', type='number', value=45, step=5),
        html.Div(dcc.Slider(id='theta2', min=0, max=180, step=5, value=45, marks={0: '0', 180: '180'}), style={'width': '50%'}),
    ]),

    # Output Graph
    dcc.Graph(id='magnetic-field-graph')
])

# Define the callback function for updating the graph
@app.callback(
    Output('magnetic-field-graph', 'figure'),
    [Input("mag-mode-1", "value"),
     Input('B1', 'value'), Input('f1', 'value'), Input('alpha1', 'value'),
     Input('beta1', 'value'), Input('gamma1', 'value'), Input('epsilon1', 'value'), Input('theta1', 'value'),
     Input("mag-mode-2", "value"),
     Input('B2', 'value'), Input('f2', 'value'), Input('alpha2', 'value'),
     Input('beta2', 'value'), Input('gamma2', 'value'), Input('epsilon2', 'value')], Input('theta2', 'value'),
)
def update_graph(mag_mode_1, B1, f1, alpha1, beta1, gamma1, epsilon1, theta1, mag_mode_2, B2, f2, alpha2, beta2, gamma2, epsilon2, theta2):
    t = np.linspace(0, 1, 500)
    # Call your compute_signals function with the updated parameters
    B_x1, B_y1, B_z1, B_x2, B_y2, B_z2 = compute_signals(mag_mode_1, mag_mode_2, B1, B2, f1, f2, alpha1, beta1, gamma1, epsilon1, theta1, alpha2, beta2, gamma2, epsilon2, theta2)
    B_length = B1 + B2

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("B1", "B2", "B Total")
    )

    # First field visualization
    fig.add_trace(
        go.Scatter3d(
            x=B_x1, y=B_y1, z=B_z1,
            mode='lines' if mag_mode_1 != 'Static' else 'markers',
            line=dict(width=6, color='red') if mag_mode_1 != 'Static' else None,
            marker=dict(size=6, color='red') if mag_mode_1 == 'Static' else None,
            name='B1'
        ),
        row=1, col=1
    )

    # Second field visualization
    fig.add_trace(
        go.Scatter3d(
            x=B_x2, y=B_y2, z=B_z2,
            mode='lines' if mag_mode_2 != 'Static' else 'markers',
            line=dict(width=6, color='green') if mag_mode_2 != 'Static' else None,
            marker=dict(size=6, color='green') if mag_mode_2 == 'Static' else None,
            name='B2'
        ),
        row=1, col=2
    )

    # 添加第三个子图：B_1 + B_2
    fig.add_trace(go.Scatter3d(
        x=B_x1 + B_x2, y=B_y1 + B_y2, z=B_z1 + B_z2,
        mode='lines',
        line=dict(width=6, color='blue'),
        name='B1+B2'
    ), row=1, col=3)

    # 添加原点标记
    marker_size = 6
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=marker_size, color='black'),
        name='O1'
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=marker_size, color='black'),
        name='O2'
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=marker_size, color='black'),
        name='O3'
    ), row=1, col=3)

    fig.update_layout(
        title='3D Magnetic Field Components',
        width=1600,
        height=600,
        scene=dict(aspectmode='cube'),
        scene2=dict(aspectmode='cube'),
        scene3=dict(aspectmode='cube')
    )
    # 设置各个子图的范围
    fig.update_scenes(
        xaxis=dict(range=[min(-2, -B_length), max(2, B_length)]),
        yaxis=dict(range=[min(-2, -B_length), max(2, B_length)]),
        zaxis=dict(range=[min(-2, -B_length), max(2, B_length)])
    )

    return fig

# 同步input和slider
@app.callback(
    [
        Output('B1', 'value'),
        Output('f1', 'value'),
        Output('alpha1', 'value'),
        Output('beta1', 'value'),
        Output('gamma1', 'value'),
        Output('epsilon1', 'value'),
        Output('theta1', 'value'),
        Output('B2', 'value'),
        Output('f2', 'value'),
        Output('alpha2', 'value'),
        Output('beta2', 'value'),
        Output('gamma2', 'value'),
        Output('epsilon2', 'value'),
        Output('theta2', 'value')
    ],
    [
        Input('B1-input', 'value'),
        Input('f1-input', 'value'),
        Input('alpha1-input', 'value'),
        Input('beta1-input', 'value'),
        Input('gamma1-input', 'value'),
        Input('epsilon1-input', 'value'),
        Input('theta1-input', 'value'),
        Input('B2-input', 'value'),
        Input('f2-input', 'value'),
        Input('alpha2-input', 'value'),
        Input('beta2-input', 'value'),
        Input('gamma2-input', 'value'),
        Input('epsilon2-input', 'value'),
        Input('theta2-input', 'value')
    ],
    prevent_initial_call=True
)
def update_sliders(B1, f1, alpha1, beta1, gamma1, epsilon1, theta1, B2, f2, alpha2, beta2, gamma2, epsilon2, theta2):
    return B1, f1, alpha1, beta1, gamma1, epsilon1, theta1, B2, f2, alpha2, beta2, gamma2, epsilon2, theta2

# Callback to update input boxes when sliders are moved
@app.callback(
    [
        Output('B1-input', 'value'),
        Output('f1-input', 'value'),
        Output('alpha1-input', 'value'),
        Output('beta1-input', 'value'),
        Output('gamma1-input', 'value'),
        Output('epsilon1-input', 'value'),
        Output('theta1-input', 'value'),
        Output('B2-input', 'value'),
        Output('f2-input', 'value'),
        Output('alpha2-input', 'value'),
        Output('beta2-input', 'value'),
        Output('gamma2-input', 'value'),
        Output('epsilon2-input', 'value'),
        Output('theta2-input', 'value')
    ],
    [
        Input('B1', 'value'),
        Input('f1', 'value'),
        Input('alpha1', 'value'),
        Input('beta1', 'value'),
        Input('gamma1', 'value'),
        Input('epsilon1', 'value'),
        Input('theta1', 'value'),
        Input('B2', 'value'),
        Input('f2', 'value'),
        Input('alpha2', 'value'),
        Input('beta2', 'value'),
        Input('gamma2', 'value'),
        Input('epsilon2', 'value'),
        Input('theta2', 'value')
    ],
    prevent_initial_call=True
)
def update_inputs(B1, f1, alpha1, beta1, gamma1, epsilon1, theta1, B2, f2, alpha2, beta2, gamma2, epsilon2, theta2):
    return B1, f1, alpha1, beta1, gamma1, epsilon1, theta1, B2, f2, alpha2, beta2, gamma2, epsilon2, theta2



# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
