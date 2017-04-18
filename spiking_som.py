import numpy as np
from brian2 import *


class ReceptiveField:

    # Parameter that used in standard deviation definition
    gamma = 1.5

    def __init__(self, bank_size=10, I_min=0.0, I_max=1.0):
        self.bank_size = bank_size
        self.field_mu = np.array([(I_min + ((2*i - 3)/2) * ((I_max - I_min)/(bank_size - 2)))
                                  for i in range(1, bank_size + 1)])
        self.field_sigma = (1.0/self.gamma) * ((I_max - I_min)/(bank_size - 2))

    def float_to_membrane_potential(self, input_vector):
        try:
            input_vector = input_vector.reshape((input_vector.shape[0], 1))
        except Exception as e:
            print("Exception: {0}\nObject shape: {1}".format(repr(e), input_vector.shape))
            exit(1)

        temp = np.exp(-((input_vector - self.field_mu) ** 2) / (2 * self.field_sigma * self.field_sigma))
        temp = temp.astype(np.float16)
        return temp


# class SpikingSOM:
#
#     def __init__(self, inst_receptive_field, prototype_dim=3):
#         self.receptive_field = inst_receptive_field
#         self.prototype_dimention = prototype_dim
#
#     def float_to_temporal_layer(self):
#
#         tau_m = 1*ms
#         u_rest = 0.0*mV
#         equation = '''
#                     dv/dt = (I_input + u_rest)/tau_m : volt
#                    '''


if __name__ == "__main__":
    rf = ReceptiveField()
    # vect = np.array([0.15, 0.07, 0.1])
    #
    # print(rf.float_to_membrane_potential(vect))
    inputs = np.array([0.15, 0.07, 0.1])
    N = 1
    # inputs = np.random.rand(3, 1)
    potential_input = rf.float_to_membrane_potential(inputs)
    potential_input = potential_input.flatten()

    teta_reset_u = -0.5
    # temporal layer
    tau_m = 1.0 * ms
    tau_r_inh2u = 1.0 * ms
    tau_f_inh2u = 5.0 * ms
    teta_u = 0.5

    equ = '''
        # inhibition connection to u layer
        ds_inh2u/dt = (-s_inh2u)/tau_r_inh2u: 1
        dw_inh2u/dt = (s_inh2u - w_inh2u)/tau_f_inh2u: 1

        # membrane potential of u layer
        dv/dt = (-v + I_ext - w_inh2u) / tau_m: 1
        I_ext : 1
    '''

    temporal_layer = NeuronGroup(N, equ, threshold='v>teta_u', reset='v=teta_reset_u', method='linear')

    temporal_layer.I_ext = 0.9
    # inhibition neuron
    tau_inh = 0.5 * ms

    tau_r_exc = 0.4 * ms
    tau_f_exc = 2.0 * ms

    tau_r_inh = 0.2 * ms
    tau_f_inh = 1.0 * ms

    teta_u_inh = 0.01
    teta_reset_inh = -0.1

    inh_equ = '''
        # inhibition connection
        ds_inh/dt = (-s_inh)/tau_r_inh: 1
        dw_inh/dt = (s_inh - w_inh)/tau_f_inh: 1

        # excitation connection
        ds_exc/dt = (-s_exc)/tau_r_exc: 1
        dw_exc/dt = (s_exc - w_exc)/tau_f_exc: 1

        # membrane potential of inhibition neuron
        dv/dt = (-v + w_exc - w_inh) / tau_inh: 1
    '''
    inhibition_neuron = NeuronGroup(1, inh_equ, threshold='v>teta_u_inh', reset='v=teta_reset_inh', method='linear')

    # v to inh neuron, excitation connection
    v2inh_excitation = Synapses(temporal_layer, target=inhibition_neuron, method='linear', on_pre="s_exc += 1")
    v2inh_excitation.connect(i=np.arange(N), j=0)

    # v to inh neuron, inhibition connection
    v2inh_inhibition = Synapses(temporal_layer, target=inhibition_neuron, method='linear', on_pre="s_inh += 1")
    v2inh_inhibition.connect(i=np.arange(N), j=0)

    # inh neuron to v, inhibition connection
    inh2v_inhibition = Synapses(inhibition_neuron, target=temporal_layer, method='linear', on_pre="s_inh2u += 1")
    inh2v_inhibition.connect(i=0, j=np.arange(N))

    u_spike_mon = SpikeMonitor(temporal_layer)
    u_state_mon_v = StateMonitor(temporal_layer, 'v', record=True)
    u_state_mon_w = StateMonitor(temporal_layer, 'w_inh2u', record=True)

    inh_spike_mon = SpikeMonitor(inhibition_neuron)
    inh_state_mon = StateMonitor(inhibition_neuron, 'v', record=True)
    w_exc_neu_state = StateMonitor(inhibition_neuron, 'w_exc', record=True)
    w_inh_neu_state = StateMonitor(inhibition_neuron, 'w_inh', record=True)

    defaultclock.dt = 0.1*ms

    simulation_time = 4
    step = 0.2

    run(simulation_time * ms, report='text')

    subplot(321)
    title("Temporal layer spikes")
    plot(u_spike_mon.t / ms, u_spike_mon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    grid(True)
    xticks(np.arange(0.0, simulation_time + step, step))
    yticks(np.arange(0, N+1, 1))

    subplot(322)
    title("Inhibition neuron spikes")
    plot(inh_spike_mon.t / ms, inh_spike_mon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    grid(True)
    xticks(np.arange(0.0, simulation_time + step, step))
    yticks(np.arange(-1, 1, 1))

    subplot(323)
    title("u membrane potential")
    plot(u_state_mon_v.t / ms, u_state_mon_v[0].v) #, u_state_mon_v.t / ms, u_state_mon_v[1].v,
         # u_state_mon_v.t / ms, u_state_mon_v[2].v, u_state_mon_v.t / ms, u_state_mon_v[3].v)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    subplot(324)
    title("Inhibition neuron membrane potential")
    plot(inh_state_mon.t / ms, inh_state_mon[0].v)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    subplot(325)
    title("Excitation/inhibition interaction")
    plot(w_exc_neu_state.t / ms, w_exc_neu_state[0].w_exc, w_exc_neu_state.t / ms, w_inh_neu_state[0].w_inh)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    subplot(326)
    title("Inhibition to u potential")
    plot(u_state_mon_w.t / ms, u_state_mon_w[0].w_inh2u)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    show()


