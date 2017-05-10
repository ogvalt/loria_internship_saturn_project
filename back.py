from flask import Flask, render_template, jsonify, request, redirect
from spiking_som import *
import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/temporal_spikes', methods=['GET', 'POST'])
def temporal_layer_spikes_get():
    time = np.around(model.u_spike_mon.t[:] / ms, decimals=5)
    spike = np.around(model.u_spike_mon.i[:], decimals=5)
    return jsonify({'x': time.tolist(), 'y': spike.tolist()})


@app.route('/som_spike', methods=['GET', 'POST'])
def som_layer_spike_get():
    time = np.around(model.som_spike_mon.t[:] / ms, decimals=5)
    spike = np.around(model.som_spike_mon.i[:], decimals=5)
    return jsonify({'x': time.tolist(), 'y': spike.tolist()})


@app.route('/inh_spike', methods=['GET', 'POST'])
def inh_neuron_spike_get():
    time = np.around(model.inh_spike_mon.t[:] / ms, decimals=5)
    spike = np.around(model.inh_spike_mon.i[:], decimals=5)
    return jsonify({'x': time.tolist(), 'y': spike.tolist()})


@app.route('/membrane_potential_temporal_layer', methods=['GET', 'POST'])
def potential_temporal_layer_get():
    print(request)
    time = np.around(model.u_state_mon_v.t[:] / ms, decimals=5)
    potential = np.around(model.u_state_mon_v[int(request.args['number'])].v[:], decimals=5)
    time_list = ['time']
    time_list.extend(time.tolist())
    temp_list = [time_list]
    neuron_list = ['neuron ' + request.args['number']]
    neuron_list.extend(potential.tolist())
    temp_list.append(neuron_list)
    return jsonify({'data': temp_list})


@app.route('/model_parameter', methods=['POST'])
def model_param():
    global model
    model = SpikingSOM()
    model.time_step = float(request.form['time_step'])
    model.tau_m = float(request.form['tau_m']) * ms
    model.tau_m_inh = float(request.form['tau_m_inh']) * ms
    model.tau_m_som = float(request.form['tau_m_som']) * ms
    model.theta_reset_u = float(request.form['theta_reset_u'])
    model.theta_reset_inh = float(request.form['theta_reset_inh'])
    model.theta_reset_som = float(request.form['theta_reset_som'])
    model.theta_u = float(request.form['theta_u'])
    model.theta_u_inh = float(request.form['theta_u_inh'])
    model.theta_som = float(request.form['theta_som'])
    # (B) Synaptic parameters, used in (2) and (3) for different synapse types
    # temporal layer to som layer (u to v)
    model.tau_r_afferent = float(request.form['tau_r_afferent']) * ms
    model.tau_f_afferent = float(request.form['tau_f_afferent']) * ms
    # temporal layer (u to inh exc, u to inh inh, inh to u)
    model.tau_r_exc = float(request.form['tau_r_exc']) * ms
    model.tau_f_exc = float(request.form['tau_f_exc']) * ms
    model.tau_r_inh = float(request.form['tau_r_inh']) * ms
    model.tau_f_inh = float(request.form['tau_f_inh']) * ms
    model.tau_r_inh2u = float(request.form['tau_r_inh2u']) * ms
    model.tau_f_inh2u = float(request.form['tau_f_inh2u']) * ms
    # som layer
    model.tau_r_lateral = float(request.form['tau_r_lateral']) * ms
    model.tau_f_lateral = float(request.form['tau_f_lateral']) * ms
    # (C) Maximum magnitudes of synaptic connection strength
    model.w_syn_temporal_to_som_max = float(request.form['w_syn_temporal_to_som_max'])
    model.w_syn_u2inh_exc_max = float(request.form['w_syn_u2inh_exc_max'])
    model.w_syn_u2inh_inh_max = float(request.form['w_syn_u2inh_inh_max']    )
    model.w_syn_inh2u_max = float(request.form['w_syn_inh2u_max'])
    model.w_syn_som_to_som_max = float(request.form['w_syn_som_to_som_max'])
    # (D) Neighbourhood parameters, used in (6) and (7), for layer v (som)
    model.a = float(request.form['a'])
    model.b = float(request.form['b'])
    model.X = float(request.form['X'])
    model.X_ = float(request.form['X_'])
    # (E) Learning parameter, used in (5)
    # A_plus - Max synaptic strength, A_minus - max synaptic weakness; tau_plus, tau_minus - time constant of STDP
    model.A_plus = float(request.form['A_plus'])
    model.A_minus = float(request.form['A_minus'])
    model.tau_plus = float(request.form['tau_plus'])
    model.tau_minus = float(request.form['tau_minus'])
    # used in (7)
    model.T = float(request.form['T'])
    model.power_n = float(request.form['power_n'])
    # size of the self-organizing map
    model.map_size = float(request.form['map_size'])
    model.simulation_time = float(request.form['simulation_time'])
    model.run_simulation()
    return jsonify("Ok!")

if __name__ == '__main__':
    app.run()