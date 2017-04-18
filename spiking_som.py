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

        return np.exp(-((input_vector - self.field_mu) ** 2) / (2 * self.field_sigma * self.field_sigma))


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
    # vect = np.random.rand(5, 1)
    #
    # print(rf.float_to_membrane_potential(vect))
    inputs = np.array([0.1, 0.1, 0.1])
    # inputs = np.random.rand(3, 1)
    potential_input = rf.float_to_membrane_potential(inputs)
    potential_input = potential_input.flatten()

    teta_reset = 0.0
    # temporal layer
    tau_m = 1.0 * ms
    teta_u = 0.5
    equ = '''
        dv/dt = (-v + I_ext) / tau_m: 1
        I_ext : 1
    '''
    temporal_layer = NeuronGroup(30, equ, threshold='v>teta_u', reset='v=teta_reset', method='linear')
    temporal_layer.I_ext = potential_input
    # inhibition neuron
    tau_inh = 0.5 * ms
    teta_u_inh = 0.01
    inh_equ = '''
        dv/dt = (-v + I_ext) / tau_inh: 1
        I_ext : 1
    '''
    inhibition_neuron = NeuronGroup(1, inh_equ, threshold='v>teta_u_inh', reset='v=teta_reset', method='linear')
    # v to inh neuron, excitation connection
    tau_r = 0.4 * ms
    tau_f = 2.0 * ms
    psp_transmit_equ = '''
        ds1/dt = (-s1)/tau_r: 1 (clock-driven)
        dw/dt = (s1 - w)/tau_f: 1 (clock-driven)
    '''
    v2inh_excitation = Synapses(temporal_layer, target=inhibition_neuron, method='linear',
                                model=psp_transmit_equ, on_pre="s1 += 1")
    v2inh_excitation.connect(i=np.arange(30), j=0)
    inhibition_neuron.I_ext = np.sum(v2inh_excitation.w)

    # v to inh neuron, inhibition connection
    # v2inh_inhibition
    # inh neuron to v, inhibition connection
    # int2v_inhibition

    s_mon = SpikeMonitor(temporal_layer)
    inh_mon = StateMonitor(inhibition_neuron, 'v', record=0)
    m = StateMonitor(v2inh_excitation, 'w', record=True)
    defaultclock.dt = 0.1*ms

    run(10 * ms, report='text')
    subplot(131)
    plot(s_mon.t / ms, s_mon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    subplot(132)
    plot(inh_mon.t / ms, inh_mon[0].v)
    xlabel('Time (ms)')
    ylabel('Potential')
    subplot(133)
    plot(m.t / ms, m[2].w)
    xlabel('Time (ms)')
    ylabel('Potential')
    show()


