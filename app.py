### DASH
from dash import Dash, dcc, html, Output, Input
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import namedtuple
from dash.dependencies import ClientsideFunction

# 初始化 Dash 应用
app = Dash(__name__)

# 定义磁场参数
MagParam= namedtuple('MagParam', ['B', 'alpha', 'beta', 'gamma', 'f', 'epsilon','theta_sweep'])

# 设置初始参数
t = np.linspace(0, 1, 500)


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
            function compute_signals(mag_mode_1, mag_mode_2, B1, B2, f1, f2, alpha1, beta1, gamma1, epsilon1, theta1, alpha2, beta2, gamma2, epsilon2, theta2) {
                const radians = (degrees) => degrees * Math.PI / 180;

                // Convert angles to radians
                alpha1 = radians(alpha1);
                beta1 = radians(beta1);
                gamma1 = radians(gamma1);
                theta1 = radians(theta1);
                alpha2 = radians(alpha2);
                beta2 = radians(beta2);
                gamma2 = radians(gamma2);
                theta2 = radians(theta2);

                let t = Array.from({ length: 500 }, (_, i) => i / 500);
                let B_x1 = [], B_y1 = [], B_z1 = [];
                let B_x2 = [], B_y2 = [], B_z2 = [];

                // First signal calculations
                if (mag_mode_1 === 'Ellipse') {
                    const phi_x1 = Math.atan2(
                        -Math.cos(gamma1) * Math.sin(alpha1) + Math.cos(alpha1) * Math.sin(beta1) * Math.sin(gamma1),
                        epsilon1 * (Math.cos(alpha1) * Math.cos(gamma1) * Math.sin(beta1) + Math.sin(alpha1) * Math.sin(gamma1))
                    );

                    const phi_y1 = Math.atan2(
                        Math.cos(alpha1) * Math.cos(gamma1) + Math.sin(alpha1) * Math.sin(beta1) * Math.sin(gamma1),
                        epsilon1 * (Math.cos(gamma1) * Math.sin(alpha1) * Math.sin(beta1) - Math.cos(alpha1) * Math.sin(gamma1))
                    );

                    const phi_z1 = Math.atan2(
                        Math.cos(beta1) * Math.sin(gamma1),
                        epsilon1 * Math.cos(beta1) * Math.cos(gamma1)
                    );

                    for (let i = 0; i < t.length; i++) {
                        B_x1[i] = B1 * Math.sqrt(
                            Math.pow(-Math.cos(gamma1) * Math.sin(alpha1) + Math.cos(alpha1) * Math.sin(beta1) * Math.sin(gamma1), 2) +
                            Math.pow(epsilon1 * (Math.cos(alpha1) * Math.cos(gamma1) * Math.sin(beta1) + Math.sin(alpha1) * Math.sin(gamma1)), 2)
                        ) * Math.sin(2 * Math.PI * f1 * t[i] + phi_x1);

                        B_y1[i] = B1 * Math.sqrt(
                            Math.pow(Math.cos(alpha1) * Math.cos(gamma1) + Math.sin(alpha1) * Math.sin(beta1) * Math.sin(gamma1), 2) +
                            Math.pow(epsilon1 * (Math.cos(gamma1) * Math.sin(alpha1) * Math.sin(beta1) - Math.cos(alpha1) * Math.sin(gamma1)), 2)
                        ) * Math.sin(2 * Math.PI * f1 * t[i] + phi_y1);

                        B_z1[i] = B1 * Math.sqrt(
                            Math.pow(Math.cos(beta1) * Math.sin(gamma1), 2) +
                            Math.pow(epsilon1 * Math.cos(beta1) * Math.cos(gamma1), 2)
                        ) * Math.sin(2 * Math.PI * f1 * t[i] + phi_z1);
                    }
                } else if (mag_mode_1 === 'Sweep') {
                    const Tri_wave = t.map(x => 4 * theta1 * Math.abs(x * f1 - Math.floor(x * f1 + 0.75) + 0.25) - theta1);

                    const phi_x1 = Math.atan2(Math.cos(alpha1) * Math.cos(beta1),
                        -Math.cos(gamma1) * Math.sin(alpha1) + Math.cos(alpha1) * Math.sin(beta1) * Math.sin(gamma1));
                    const phi_y1 = Math.atan2(Math.cos(beta1) * Math.sin(alpha1),
                        Math.cos(alpha1) * Math.cos(gamma1) + Math.sin(alpha1) * Math.sin(beta1) * Math.sin(gamma1));
                    const phi_z1 = Math.atan2(-Math.sin(beta1), Math.cos(beta1) * Math.sin(gamma1));

                    for (let i = 0; i < t.length; i++) {
                        B_x1[i] = B1 * Math.sqrt(
                            Math.pow(Math.cos(alpha1) * Math.cos(beta1), 2) +
                            Math.pow(-Math.cos(gamma1) * Math.sin(alpha1) + Math.cos(alpha1) * Math.sin(beta1) * Math.sin(gamma1), 2)
                        ) * Math.sin(Tri_wave[i] + phi_x1);

                        B_y1[i] = B1 * Math.sqrt(
                            Math.pow(Math.cos(beta1) * Math.sin(alpha1), 2) +
                            Math.pow(Math.cos(alpha1) * Math.cos(gamma1) + Math.sin(alpha1) * Math.sin(beta1) * Math.sin(gamma1), 2)
                        ) * Math.sin(Tri_wave[i] + phi_y1);

                        B_z1[i] = B1 * Math.sqrt(
                            Math.pow(Math.sin(beta1), 2) +
                            Math.pow(Math.cos(beta1) * Math.sin(gamma1), 2)
                        ) * Math.sin(Tri_wave[i] + phi_z1);
                    }
                } else if (mag_mode_1 === 'Static') {
                    for (let i = 0; i < t.length; i++) {
                        B_x1[i] = B1 * Math.cos(alpha1) * Math.cos(beta1);
                        B_y1[i] = B1 * Math.cos(beta1) * Math.sin(alpha1);
                        B_z1[i] = -B1 * Math.sin(beta1);
                    }
                }

                // Second signal calculations (similar to the first one)
                if (mag_mode_2 === 'Ellipse') {
                    const phi_x2 = Math.atan2(
                        Math.cos(alpha2) * Math.sin(beta2) + Math.sin(alpha2) * Math.cos(beta2) * Math.sin(gamma2),
                        epsilon2 * (Math.cos(alpha2) * Math.cos(beta2) - Math.sin(alpha2) * Math.sin(beta2) * Math.sin(gamma2))
                    );

                    const phi_y2 = Math.atan2(
                        Math.sin(alpha2) * Math.sin(beta2) - Math.cos(alpha2) * Math.cos(beta2) * Math.sin(gamma2),
                        epsilon2 * (Math.sin(alpha2) * Math.cos(beta2) + Math.cos(alpha2) * Math.sin(beta2) * Math.sin(gamma2))
                    );

                    const phi_z2 = Math.atan2(
                        Math.cos(beta2) * Math.cos(gamma2),
                        -epsilon2 * Math.sin(beta2) * Math.cos(gamma2)
                    );

                    for (let i = 0; i < t.length; i++) {
                        B_x2[i] = B2 * Math.sqrt(
                            Math.pow(Math.cos(alpha2) * Math.sin(beta2) + Math.sin(alpha2) * Math.cos(beta2) * Math.sin(gamma2), 2) +
                            Math.pow(epsilon2 * (Math.cos(alpha2) * Math.cos(beta2) - Math.sin(alpha2) * Math.sin(beta2) * Math.sin(gamma2)), 2)
                        ) * Math.sin(2 * Math.PI * f2 * t[i] + phi_x2);

                        B_y2[i] = B2 * Math.sqrt(
                            Math.pow(Math.sin(alpha2) * Math.sin(beta2) - Math.cos(alpha2) * Math.cos(beta2) * Math.sin(gamma2), 2) +
                            Math.pow(epsilon2 * (Math.sin(alpha2) * Math.cos(beta2) + Math.cos(alpha2) * Math.sin(beta2) * Math.sin(gamma2)), 2)
                        ) * Math.sin(2 * Math.PI * f2 * t[i] + phi_y2);

                        B_z2[i] = B2 * Math.sqrt(
                            Math.pow(Math.cos(beta2) * Math.cos(gamma2), 2) +
                            Math.pow(-epsilon2 * Math.sin(beta2) * Math.cos(gamma2), 2)
                        ) * Math.sin(2 * Math.PI * f2 * t[i] + phi_z2);
                    }
                } else if (mag_mode_2 === 'Sweep') {
                    const Tri_wave = t.map(x => 4 * theta2 * Math.abs(x * f2 - Math.floor(x * f2 + 0.75) + 0.25) - theta2);

                    const phi_x2 = Math.atan2(-Math.sin(alpha2) * Math.cos(gamma2),
                        Math.cos(alpha2) * Math.sin(beta2) + Math.sin(alpha2) * Math.cos(beta2) * Math.sin(gamma2));
                    const phi_y2 = Math.atan2(Math.cos(alpha2) * Math.cos(gamma2),
                        Math.sin(alpha2) * Math.sin(beta2) - Math.cos(alpha2) * Math.cos(beta2) * Math.sin(gamma2));
                    const phi_z2 = Math.atan2(Math.sin(gamma2), Math.cos(beta2) * Math.cos(gamma2));

                    for (let i = 0; i < t.length; i++) {
                        B_x2[i] = B2 * Math.sqrt(
                            Math.pow(-Math.sin(alpha2) * Math.cos(gamma2), 2) +
                            Math.pow(Math.cos(alpha2) * Math.sin(beta2) + Math.sin(alpha2) * Math.cos(beta2) * Math.sin(gamma2), 2)
                        ) * Math.sin(Tri_wave[i] + phi_x2);
                        B_y2[i] = B2 * Math.sqrt(
                            Math.pow(Math.cos(alpha2) * Math.cos(gamma2), 2) +
                            Math.pow(Math.sin(alpha2) * Math.sin(beta2) - Math.cos(alpha2) * Math.cos(beta2) * Math.sin(gamma2), 2)
                        ) * Math.sin(Tri_wave[i] + phi_y2);
                        B_z2[i] = B2 * Math.sqrt(
                            Math.pow(Math.sin(gamma2), 2) +
                            Math.pow(Math.cos(beta2) * Math.cos(gamma2), 2)
                        ) * Math.sin(Tri_wave[i] + phi_z2);
                    }
                } else if (mag_mode_2 === 'Static') {
                    for (let i = 0; i < t.length; i++) {
                        B_x2[i] = -B2 * Math.sin(alpha2) * Math.cos(gamma2);
                        B_y2[i] = B2 * Math.cos(alpha2) * Math.cos(gamma2);
                        B_z2[i] = B2 * Math.sin(gamma2);
                    }
                }

                // Return the computed signals
                return [B_x1, B_y1, B_z1, B_x2, B_y2, B_z2];
            }
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


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
app.clientside_callback(
    """
    function(mag_mode_1, B1, f1, alpha1, beta1, gamma1, epsilon1, theta1,
             mag_mode_2, B2, f2, alpha2, beta2, gamma2, epsilon2, theta2) {
        var t = [];
        for (var i = 0; i < 500; i++) {
            t.push(i / 500);
        }

        // Call your compute_signals function here
        var signals = compute_signals(mag_mode_1, mag_mode_2, B1, B2, f1, f2, alpha1, beta1, gamma1, epsilon1, theta1, alpha2, beta2, gamma2, epsilon2, theta2);
        var B_x1 = signals[0], B_y1 = signals[1], B_z1 = signals[2];
        var B_x2 = signals[3], B_y2 = signals[4], B_z2 = signals[5];

        var traces = [];

        // First field visualization
        traces.push({
            x: B_x1,
            y: B_y1,
            z: B_z1,
            mode: mag_mode_1 !== 'Static' ? 'lines' : 'markers',
            type: 'scatter3d',
            line: { width: 6, color: 'red' },
            marker: { size: 6, color: 'red' },
            name: 'B1'
        });

        // Second field visualization
        traces.push({
            x: B_x2,
            y: B_y2,
            z: B_z2,
            mode: mag_mode_2 !== 'Static' ? 'lines' : 'markers',
            type: 'scatter3d',
            line: { width: 6, color: 'green' },
            marker: { size: 6, color: 'green' },
            name: 'B2'
        });

        // Third subplot: B1 + B2
        traces.push({
            x: B_x1.map((v, i) => v + B_x2[i]),
            y: B_y1.map((v, i) => v + B_y2[i]),
            z: B_z1.map((v, i) => v + B_z2[i]),
            mode: 'lines',
            type: 'scatter3d',
            line: { width: 6, color: 'blue' },
            name: 'B1+B2'
        });

        // Adding initial vectors
        traces.push({
            x: [0, B_x1[0]], y: [0, B_y1[0]], z: [0, B_z1[0]],
            mode: 'line',
            type: 'scatter3d',
            marker: { size: 6, color: 'rgb(251, 180, 174)' },
            name: 'Initial B1'
        });
        
        traces.push({
            x: [0, B_x2[0]], y: [0, B_y2[0]], z: [0, B_z2[0]],
            mode: 'line',
            type: 'scatter3d',
            marker: { size: 6, color: 'rgb(204, 235, 197)' },
            name: 'Initial B2'
        });
        
        traces.push({
            x: [0, B_x1[0]+B_x2[0]], y: [0, B_y1[0]+B_y2[0]], z: [0, B_z1[0]+B_z2[0]],
            mode: 'line',
            type: 'scatter3d',
            marker: { size: 6, color: 'rgb(179, 205, 227)' },
            name: 'Initial B'
        });

        // Adding origin markers
        traces.push({
            x: [0], y: [0], z: [0],
            mode: 'markers',
            type: 'scatter3d',
            marker: { size: 6, color: 'black' },
            name: 'O1'
        });

        traces.push({
            x: [0], y: [0], z: [0],
            mode: 'markers',
            type: 'scatter3d',
            marker: { size: 6, color: 'black' },
            name: 'O2'
        });

        traces.push({
            x: [0], y: [0], z: [0],
            mode: 'markers',
            type: 'scatter3d',
            marker: { size: 6, color: 'black' },
            name: 'O3'
        });

        return {
            data: traces,
            layout: {
                title: '3D Magnetic Field Components',
                width: 1600,
                height: 600,
                scene: {
                    aspectmode: 'cube',
                    xaxis: { title: 'X', range: [Math.min(-2, -B1-B2), Math.max(2, B1+B2)] },
                    yaxis: { title: 'Y', range: [Math.min(-2, -B1-B2), Math.max(2, B1+B2)] },
                    zaxis: { title: 'Z', range: [Math.min(-2, -B1-B2), Math.max(2, B1+B2)] }
                }
            }
        };
    }
    """,
    Output('magnetic-field-graph', 'figure'),
    [
        Input("mag-mode-1", "value"),
        Input('B1', 'value'), Input('f1', 'value'), Input('alpha1', 'value'),
        Input('beta1', 'value'), Input('gamma1', 'value'), Input('epsilon1', 'value'), Input('theta1', 'value'),
        Input("mag-mode-2", "value"),
        Input('B2', 'value'), Input('f2', 'value'), Input('alpha2', 'value'),
        Input('beta2', 'value'), Input('gamma2', 'value'), Input('epsilon2', 'value'), Input('theta2', 'value'),
    ]
)


# 同步input和slider
app.clientside_callback(
    """
    function(B1, f1, alpha1, beta1, gamma1, epsilon1, theta1,
             B2, f2, alpha2, beta2, gamma2, epsilon2, theta2) {
        // Return the values directly
        return [B1, f1, alpha1, beta1, gamma1, epsilon1, theta1,
                B2, f2, alpha2, beta2, gamma2, epsilon2, theta2];
    }
    """,
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


app.clientside_callback(
    """
    function(B1, f1, alpha1, beta1, gamma1, epsilon1, theta1,
             B2, f2, alpha2, beta2, gamma2, epsilon2, theta2) {
        // Return the values directly
        return [B1, f1, alpha1, beta1, gamma1, epsilon1, theta1,
                B2, f2, alpha2, beta2, gamma2, epsilon2, theta2];
    }
    """,
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



# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
