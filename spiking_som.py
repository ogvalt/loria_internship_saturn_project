import numpy as np
from brian2 import *


class ReceptiveField:
    # Parameter that used in standard deviation definition
    gamma = 1.5

    def __init__(self, bank_size=10, I_min=0.0, I_max=1.0):
        self.bank_size = bank_size
        self.field_mu = np.array([(I_min + ((2 * i - 2) / 2) * ((I_max - I_min) / (bank_size - 1)))
                                  for i in range(1, bank_size + 1)])
        self.field_sigma = (1.0 / self.gamma) * (I_max - I_min)

    def float_to_membrane_potential(self, input_vector):
        try:
            input_vector = input_vector.reshape((input_vector.shape[0], 1))
        except Exception as exc:
            print("Exception: {0}\nObject shape: {1}".format(repr(exc), input_vector.shape))
            exit(1)

        temp = np.exp(-((input_vector - self.field_mu) ** 2) / (2 * self.field_sigma * self.field_sigma)) / \
               (np.sqrt(2 * np.pi) * self.field_sigma)
        temp += np.exp(-((input_vector - 1 - self.field_mu) ** 2) / (2 * self.field_sigma * self.field_sigma)) / \
                (np.sqrt(2 * np.pi) * self.field_sigma)
        temp += np.exp(-((input_vector + 1 - self.field_mu) ** 2) / (2 * self.field_sigma * self.field_sigma)) / \
                (np.sqrt(2 * np.pi) * self.field_sigma)

        return temp


class SpikingSOM:
    def __init__(self):
        # (A) Neuronal parameters, used in (1) and (4)
        self.time_step = 0.01
        self.tau_m = 1.0 * ms
        self.tau_m_inh = 0.5 * ms
        self.tau_m_som = 1.0 * ms
        self.theta_reset_u = -0.5
        self.theta_reset_inh = -0.1
        self.theta_reset_som = 0.0
        self.theta_u = 0.5
        self.theta_u_inh = 0.01
        self.theta_som = 1.0
        # (B) Synaptic parameters, used in (2) and (3) for different synapse types
        # temporal layer to som layer (u to v)
        self.tau_r_afferent = 0.2 * ms
        self.tau_f_afferent = 1.0 * ms
        # temporal layer (u to inh exc, u to inh inh, inh to u)
        self.tau_r_exc = 0.4 * ms
        self.tau_f_exc = 2.0 * ms
        self.tau_r_inh = 0.2 * ms
        self.tau_f_inh = 1.0 * ms
        self.tau_r_inh2u = 1.0 * ms
        self.tau_f_inh2u = 5.0 * ms
        # som layer
        self.tau_r_lateral = 0.1 * ms
        self.tau_f_lateral = 0.5 * ms
        # (C) Maximum magnitudes of synaptic connection strength
        self.w_syn_temporal_to_som_max = 2.2
        self.w_syn_u2inh_exc_max = 1.0
        self.w_syn_u2inh_inh_max = 1.0
        self.w_syn_inh2u_max = 100.0
        self.w_syn_som_to_som_max = 1.0
        # (D) Neighbourhood parameters, used in (6) and (7), for layer v (som)
        self.a = 3.0
        self.b = 3.0
        self.X = 3.0
        self.X_ = 3.0
        # (E) Learning parameter, used in (5)
        # A_plus - Max synaptic strength, A_minus - max synaptic weakness; tau_plus, tau_minus - time constant of STDP
        self.A_plus = 0.0016
        self.A_minus = 0.0055
        self.tau_plus = 11
        self.tau_minus = 10

        # used in (7)
        self.T = 1.0
        self.power_n = 1.0
        # size of the self-organizing map
        self.map_size = 10
        self.simulation_time = 4

        self.som_spike_mon = None
        self.u_spike_mon = None
        self.inh_spike_mon = None
        self.u_state_mon_v = None
        self.u_state_mon_time = None
        self.u_state_mon_w = None
        self.inh_state_mon = None
        self.w_exc_neu_state = None
        self.w_inh_neu_state = None
        self.w_syn_u2inh_exc = None
        self.w_syn_u2inh_inh = None
        self.w_syn_u2inh = None
        self.som_state_mon_v = None
        self.som_state_mon_w_afferent = None
        self.som_state_mon_w_lateral = None
        self.w_syn_u2v_state = None

    def run_simulation(self):
        np.random.seed(1)
        seed(1)
        np.set_printoptions(suppress=True)

        bank_size = 10
        diff_method = 'euler'

        inputs = np.random.rand(3)
        print(inputs)
        N = inputs.shape[0] * bank_size

        rf = ReceptiveField(bank_size=bank_size, I_min=0.05, I_max=0.95)
        potential_input = rf.float_to_membrane_potential(inputs)
        potential_input = potential_input.flatten()

        # TABLE 1
        # (A) Neuronal parameters, used in (1) and (4)
        time_step = self.time_step
        tau_m = self.tau_m
        tau_m_inh = self.tau_m_inh
        tau_m_som = self.tau_m_som
        theta_reset_u = self.theta_reset_u
        theta_reset_inh = self.theta_reset_inh
        theta_reset_som = self.theta_reset_som
        theta_u = self.theta_u
        theta_u_inh = self.theta_u_inh
        theta_som = self.theta_som
        # (B) Synaptic parameters, used in (2) and (3) for different synapse types
        # temporal layer to som layer (u to v)
        tau_r_afferent = self.tau_r_afferent
        tau_f_afferent = self.tau_f_afferent
        # temporal layer (u to inh exc, u to inh inh, inh to u)
        tau_r_exc = self.tau_r_exc
        tau_f_exc = self.tau_f_exc
        tau_r_inh = self.tau_r_inh
        tau_f_inh = self.tau_f_inh
        tau_r_inh2u = self.tau_r_inh2u
        tau_f_inh2u = self.tau_f_inh2u
        # som layer
        tau_r_lateral = self.tau_r_lateral
        tau_f_lateral = self.tau_f_lateral
        # (C) Maximum magnitudes of synaptic connection strength
        w_syn_temporal_to_som_max = self.w_syn_temporal_to_som_max
        w_syn_u2inh_exc_max = self.w_syn_u2inh_exc_max
        w_syn_u2inh_inh_max = self.w_syn_u2inh_inh_max
        w_syn_inh2u_max = self.w_syn_inh2u_max
        w_syn_som_to_som_max = self.w_syn_som_to_som_max
        # (D) Neighbourhood parameters, used in (6) and (7), for layer v (som)
        a = self.a
        b = self.b
        X = self.X
        X_ = self.X_
        # (E) Learning parameter, used in (5)
        # A_plus - Max synaptic strength, A_minus - max synaptic weakness; tau_plus, tau_minus - time constant of STDP
        A_plus = self.A_plus
        A_minus = self.A_minus
        tau_plus = self.tau_plus
        tau_minus = self.tau_minus

        # used in (7)
        T = self.T
        power_n = self.power_n
        # used in (6)
        pi = np.pi
        # size of the self-organizing map
        map_size = self.map_size
        simulation_time = self.simulation_time

        temporal_layer_neuron_equ = '''
            dtime/dt = 1 / ms : 1
            
            # inhibition connection to u layer
            
            ds_inh2u/dt = (-s_inh2u)/tau_r_inh2u: 1
            dw_inh2u/dt = (s_inh2u - w_inh2u)/tau_f_inh2u: 1

            # membrane potential of u layer
            
            dv/dt = (-v + I_ext - w_inh2u) / tau_m: 1
            I_ext : 1
            '''
        inhibition_neuron_equ = '''
                dtime/dt = 1 / ms : 1

                # inhibition connection
                # s_inh - internal variable
                # w_inh - output potential

                ds_inh/dt = (-s_inh)/tau_r_inh: 1
                dw_inh/dt = (s_inh - w_inh)/tau_f_inh: 1

                # excitation connection
                # s_exc - internal variable
                # w_exc - output potential
                ds_exc/dt = (-s_exc)/tau_r_exc: 1
                dw_exc/dt = (s_exc - w_exc)/tau_f_exc: 1

                # diff equation membrane potential of inhibition neuron
                dv/dt = (-v + w_exc - w_inh) / tau_m_inh: 1
            '''
        som_layer_neuron_equ = '''
                    dglobal_time/dt = 1 / ms : 1
                    dtime/dt = 1 / ms : 1

                    # Afferent connection (from temporal layer to som layer)

                    ds_afferent/dt = (-s_afferent)/tau_r_afferent: 1
                    dw_afferent/dt = (s_afferent - w_afferent)/tau_f_afferent: 1 

                    # lateral connection

                    ds_lateral/dt = (-s_lateral)/tau_r_lateral: 1
                    dw_lateral/dt = (s_lateral - w_lateral)/tau_f_lateral: 1

                    # membrane potential of u layer

                    dv/dt = (-v + w_lateral + w_afferent) / tau_m_som: 1
                '''

        temporal_layer = NeuronGroup(N, temporal_layer_neuron_equ, threshold='v>theta_u', method=diff_method,
                                     reset='''v = theta_reset_u; time = 0''')
        temporal_layer.I_ext = potential_input

        # inhibition neuron
        inhibition_neuron = NeuronGroup(1, inhibition_neuron_equ, threshold='v>theta_u_inh', method=diff_method,
                                        reset='''v = theta_reset_inh; time = 0''')
        # self-organizing layer
        som_layer = NeuronGroup(map_size * map_size, som_layer_neuron_equ, threshold='v>theta_som', method=diff_method,
                                reset='''v = theta_reset_som; time = 0''')

        # v to inh neuron, excitation connection
        u2inh_excitation = Synapses(temporal_layer, target=inhibition_neuron, method=diff_method,
                                    on_pre='''
                                    s_exc += w_syn
                                    A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) ** time_post
                                    w_syn = clip(w_syn + plasticity * A_pre, 0, w_syn_u2inh_exc_max) 
                                    ''',
                                    on_post='''
                                    A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) ** time_pre
                                    w_syn = clip(w_syn + plasticity * A_post, 0, w_syn_u2inh_exc_max)
                                    ''',
                                    model='''
                                    w_syn : 1 # synaptic weight / synapse efficacy
                                    plasticity : boolean (shared)
                                    ''')
        u2inh_excitation.connect(i=np.arange(N), j=0)
        u2inh_excitation.w_syn = 'rand() * w_syn_u2inh_exc_max'

        # v to inh neuron, inhibition connection
        u2inh_inhibition = Synapses(temporal_layer, target=inhibition_neuron, method=diff_method,
                                    on_pre='''
                                    s_inh += w_syn
                                    A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) * time_post
                                    w_syn = clip(w_syn + plasticity * A_pre, 0, w_syn_u2inh_inh_max) 
                                    ''',
                                    on_post='''
                                    A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) * time_pre
                                    w_syn = clip(w_syn + plasticity * A_post, 0, w_syn_u2inh_inh_max)
                                    ''',
                                    model='''
                                    w_syn : 1 # synaptic weight / synapse efficacy 
                                    plasticity : boolean (shared)
                                    ''')
        u2inh_inhibition.connect(i=np.arange(N), j=0)
        u2inh_inhibition.w_syn = 'rand() * w_syn_u2inh_inh_max'

        # inh neuron to v, inhibition connection
        inh2u_inhibition = Synapses(inhibition_neuron, target=temporal_layer, method=diff_method,
                                    on_pre='''
                                    s_inh2u += w_syn
                                    A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) * time_post
                                    w_syn = clip(w_syn + plasticity * A_pre, 0, w_syn_inh2u_max)
                                    ''',
                                    on_post='''
                                    A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) * time_pre
                                    w_syn = clip(w_syn + plasticity * A_post, 0, w_syn_inh2u_max)
                                    ''',
                                    model='''
                                    w_syn : 1 # synaptic weight / synapse efficacy
                                    plasticity : boolean (shared)
                                    ''')
        inh2u_inhibition.connect(i=0, j=np.arange(N))
        inh2u_inhibition.w_syn = 'rand() * w_syn_inh2u_max'

        # som lateral connection
        som_synapse = Synapses(som_layer, target=som_layer, method=diff_method,
                               on_pre='''
                               radius = X - (X - X_)/(1+(2**0.5 - 1)*((T/global_time)**(2 * power_n)))
                               
                               y_pre = floor(i / map_size)
                               x_pre = i - y_pre * map_size
                               
                               y_post = floor(j/map_size)
                               x_post = j - y_post * map_size
                                
                               dist = (x_post - x_pre)**2 + (y_post - y_pre)**2
                               
                               G1 = (1 + a) * exp(- dist/(radius**2)) / (2 * pi * radius**2)
                               G2 = a * exp(- dist/(b * radius)**2) / (2 * pi * (b * radius)**2)
                               
                               w_syn = clip(G1 + G2, 0, w_syn_som_to_som_max)
                               s_lateral += w_syn
                               ''',
                               on_post='''
                               ''',
                               model='''
                               w_syn : 1 # synaptic weight / synapse efficacy
                               ''')
        som_synapse.connect(condition='i!=j')
        # som afferent connection
        temporal_to_som_synapse = Synapses(temporal_layer, target=som_layer, method=diff_method,
                                           on_pre='''
                                           s_afferent += w_syn
                                           A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) ** time_post
                                           w_syn = clip(w_syn + plasticity * A_pre, 0,  w_syn_temporal_to_som_max)
                                           ''',
                                           on_post='''
                                           A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) * time_pre
                                           w_syn = clip(w_syn + plasticity * A_post, 0,  w_syn_temporal_to_som_max)
                                           ''',
                                           model='''
                                           w_syn : 1 # synaptic weight / synapse efficacy
                                           plasticity : boolean (shared)
                                           ''')
        temporal_to_som_synapse.connect()
        temporal_to_som_synapse.w_syn = np.random.randint(low=40000, high=60000, size=3000) \
                                        * w_syn_temporal_to_som_max / 100000.0

        # Visualization
        # spike monitor
        som_spike_mon = SpikeMonitor(som_layer)
        u_spike_mon = SpikeMonitor(temporal_layer)
        inh_spike_mon = SpikeMonitor(inhibition_neuron)
        # potential of temporal layer
        u_state_mon_v = StateMonitor(temporal_layer, 'v', record=True)
        u_state_mon_time = StateMonitor(temporal_layer, 'time', record=True)
        u_state_mon_w = StateMonitor(temporal_layer, 'w_inh2u', record=True)
        # inhibition neuron
        inh_state_mon = StateMonitor(inhibition_neuron, 'v', record=True)
        w_exc_neu_state = StateMonitor(inhibition_neuron, 'w_exc', record=True)
        w_inh_neu_state = StateMonitor(inhibition_neuron, 'w_inh', record=True)
        # u to inh exc synaptic strenght
        w_syn_u2inh_exc = StateMonitor(u2inh_excitation, 'w_syn', record=True)
        # u to inh inh synaptic strenght
        w_syn_u2inh_inh = StateMonitor(u2inh_inhibition, 'w_syn', record=True)
        # int to u synaptic strenght
        w_syn_u2inh = StateMonitor(inh2u_inhibition, 'w_syn', record=True)
        # potential of som layer
        som_state_mon_v = StateMonitor(som_layer, 'v', record=True)
        som_state_mon_w_afferent = StateMonitor(som_layer, 'w_afferent', record=True)
        som_state_mon_w_lateral = StateMonitor(som_layer, 'w_lateral', record=True)
        # u to v afferent connection
        w_syn_u2v_state = StateMonitor(temporal_to_som_synapse, 'w_syn', record=True)

        defaultclock.dt = time_step * ms

        plasticity_state = True
        u2inh_excitation.plasticity = plasticity_state
        u2inh_inhibition.plasticity = plasticity_state
        inh2u_inhibition.plasticity = plasticity_state
        temporal_to_som_synapse.plasticity = plasticity_state

        run(simulation_time * ms, report='text')

        self.som_spike_mon = som_spike_mon
        self.u_spike_mon = u_spike_mon
        self.inh_spike_mon = inh_spike_mon
        # potential of temporal layer
        self.u_state_mon_v = u_state_mon_v
        self.u_state_mon_time = u_state_mon_time
        self.u_state_mon_w = u_state_mon_w
        # inhibition neuron
        self.inh_state_mon = inh_state_mon
        self.w_exc_neu_state = w_exc_neu_state
        self.w_inh_neu_state = w_inh_neu_state
        # u to inh exc synaptic strenght
        self.w_syn_u2inh_exc = w_syn_u2inh_exc
        # u to inh inh synaptic strenght
        self.w_syn_u2inh_inh = w_syn_u2inh_inh
        # int to u synaptic strenght
        self.w_syn_u2inh = w_syn_u2inh
        # potential of som layer
        self.som_state_mon_v = som_state_mon_v
        self.som_state_mon_w_afferent = som_state_mon_w_afferent
        self.som_state_mon_w_lateral = som_state_mon_w_lateral
        # u to v afferent connection
        self.w_syn_u2v_state = w_syn_u2v_state


if __name__ == "__main__":
    np.random.seed(1)
    seed(1)
    np.set_printoptions(suppress=True)

    bank_size = 10
    diff_method = 'euler'

    inputs = np.random.rand(3)
    # inputs = np.array([0.332, 0.167, 0.946])
    # inputs = np.array([0.013, 0.3401, 0.2196])
    # inputs = np.array([0.829, 0.7452, 0.6728])
    print(inputs)
    N = inputs.shape[0] * bank_size

    rf = ReceptiveField(bank_size=bank_size, I_min=0.05, I_max=0.95)
    potential_input = rf.float_to_membrane_potential(inputs)
    potential_input = potential_input.flatten()

    # TABLE 1 
    # (A) Neuronal parameters, used in (1) and (4)
    time_step = 0.01;
    tau_m = 10.0 * ms;
    tau_m_inh = 5 * ms;
    tau_m_som = 3 * ms
    theta_reset_u = -0.5;
    theta_reset_inh = -0.0;
    theta_reset_som = 0.0
    theta_u = 0.5;
    theta_u_inh = 0.01;
    theta_som = 1.0
    # (B) Synaptic parameters, used in (2) and (3) for different synapse types
    # temporal layer to som layer (u to v)
    tau_r_afferent = 0.2 * ms;
    tau_f_afferent = 1.0 * ms
    # temporal layer (u to inh exc, u to inh inh, inh to u)
    tau_r_exc = 0.4 * ms;
    tau_f_exc = 2.0 * ms;
    tau_r_inh = 0.2 * ms;
    tau_f_inh = 1.0 * ms
    tau_r_inh2u = 1.0 * ms;
    tau_f_inh2u = 5.0 * ms
    # som layer
    tau_r_lateral = 0.1 * ms;
    tau_f_lateral = 0.5 * ms
    # (C) Maximum magnitudes of synaptic connection strength
    w_syn_temporal_to_som_max = 2.2;
    w_syn_u2inh_exc_max = 1.0;
    w_syn_u2inh_inh_max = 1.0;
    w_syn_inh2u_max = 100.0
    w_syn_som_to_som_max = 1.0
    # (D) Neighbourhood parameters, used in (6) and (7), for layer v (som)
    a = 3.0;
    b = 3.0;
    X = 3.0;
    X_ = 3.0
    # (E) Learning parameter, used in (5)
    # A_plus - Max synaptic strength, A_minus - max synaptic weakness; tau_plus, tau_minus - time constant of STDP
    A_plus = 0.0016;
    A_minus = 0.0055;
    tau_plus = 11;
    tau_minus = 10

    # used in (7)
    T = 1.0;
    power_n = 1.0
    # used in (6)
    pi = np.pi
    # size of the self-organizing map
    map_size = 10
    simulation_time = 100

    temporal_layer_neuron_equ = '''
        dtime/dt = 1 / ms : 1
        
        # inhibition connection to u layer
        
        ds_inh2u/dt = (-s_inh2u)/tau_r_inh2u: 1
        dw_inh2u/dt = (s_inh2u - w_inh2u)/tau_f_inh2u: 1

        # membrane potential of u layer
        
        dv/dt = (-v + I_ext - w_inh2u) / tau_m: 1
        I_ext : 1
    '''
    inhibition_neuron_equ = '''
            dtime/dt = 1 / ms : 1

            # inhibition connection
            # s_inh - internal variable
            # w_inh - output potential

            ds_inh/dt = (-s_inh)/tau_r_inh: 1
            dw_inh/dt = (s_inh - w_inh)/tau_f_inh: 1

            # excitation connection
            # s_exc - internal variable
            # w_exc - output potential
            ds_exc/dt = (-s_exc)/tau_r_exc: 1
            dw_exc/dt = (s_exc - w_exc)/tau_f_exc: 1

            # diff equation membrane potential of inhibition neuron
            dv/dt = (-v + w_exc - w_inh) / tau_m_inh: 1
        '''
    som_layer_neuron_equ = '''
                dglobal_time/dt = 1 / ms : 1
                dtime/dt = 1 / ms : 1

                # Afferent connection (from temporal layer to som layer)

                ds_afferent/dt = (-s_afferent)/tau_r_afferent: 1
                dw_afferent/dt = (s_afferent - w_afferent)/tau_f_afferent: 1 

                # lateral connection

                ds_lateral/dt = (-s_lateral)/tau_r_lateral: 1
                dw_lateral/dt = (s_lateral - w_lateral)/tau_f_lateral: 1

                # membrane potential of u layer

                dv/dt = (-v + w_lateral + w_afferent) / tau_m_som: 1
            '''

    temporal_layer = NeuronGroup(N, temporal_layer_neuron_equ, threshold='v>theta_u', method=diff_method,
                                 reset='''v = theta_reset_u; time = 0''')
    temporal_layer.I_ext = potential_input

    # inhibition neuron
    inhibition_neuron = NeuronGroup(1, inhibition_neuron_equ, threshold='v>theta_u_inh', method=diff_method,
                                    reset='''v = theta_reset_inh; time = 0''')
    # self-organizing layer
    som_layer = NeuronGroup(map_size * map_size, som_layer_neuron_equ, threshold='v>theta_som', method=diff_method,
                            reset='''v = theta_reset_som; time = 0''')

    # v to inh neuron, excitation connection
    u2inh_excitation = Synapses(temporal_layer, target=inhibition_neuron, method=diff_method,
                                on_pre='''
                                s_exc += w_syn
                                A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) ** time_post
                                w_syn = clip(w_syn + plasticity * A_pre, 0, w_syn_u2inh_exc_max) 
                                ''',
                                on_post='''
                                A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) ** time_pre
                                w_syn = clip(w_syn + plasticity * A_post, 0, w_syn_u2inh_exc_max)
                                ''',
                                model='''
                                w_syn : 1 # synaptic weight / synapse efficacy
                                plasticity : boolean (shared)
                                ''')
    u2inh_excitation.connect(i=np.arange(N), j=0)
    u2inh_excitation.w_syn = 'rand() * w_syn_u2inh_exc_max'

    # v to inh neuron, inhibition connection
    u2inh_inhibition = Synapses(temporal_layer, target=inhibition_neuron, method=diff_method,
                                on_pre='''
                                s_inh += w_syn
                                A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) * time_post
                                w_syn = clip(w_syn + plasticity * A_pre, 0, w_syn_u2inh_inh_max) 
                                ''',
                                on_post='''
                                A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) * time_pre
                                w_syn = clip(w_syn + plasticity * A_post, 0, w_syn_u2inh_inh_max)
                                ''',
                                model='''
                                w_syn : 1 # synaptic weight / synapse efficacy 
                                plasticity : boolean (shared)
                                ''')
    u2inh_inhibition.connect(i=np.arange(N), j=0)
    u2inh_inhibition.w_syn = 'rand() * w_syn_u2inh_inh_max'

    # inh neuron to v, inhibition connection
    inh2u_inhibition = Synapses(inhibition_neuron, target=temporal_layer, method=diff_method,
                                on_pre='''
                                s_inh2u += w_syn
                                A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) * time_post
                                w_syn = clip(w_syn + plasticity * A_pre, 0, w_syn_inh2u_max)
                                ''',
                                on_post='''
                                A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) * time_pre
                                w_syn = clip(w_syn + plasticity * A_post, 0, w_syn_inh2u_max)
                                ''',
                                model='''
                                w_syn : 1 # synaptic weight / synapse efficacy
                                plasticity : boolean (shared)
                                ''')
    inh2u_inhibition.connect(i=0, j=np.arange(N))
    # inh2u_inhibition.w_syn = 'rand() * w_syn_inh2u_max'
    inh2u_inhibition.w_syn = 0.5 * w_syn_inh2u_max
    # som lateral connection
    som_synapse = Synapses(som_layer, target=som_layer, method=diff_method,
                           on_pre='''
                           radius = X - (X - X_)/(1+(2**0.5 - 1)*((T/global_time)**(2 * power_n)))
                           
                           y_pre = floor(i / map_size)
                           x_pre = i - y_pre * map_size
                           
                           y_post = floor(j/map_size)
                           x_post = j - y_post * map_size
                            
                           dist = (x_post - x_pre)**2 + (y_post - y_pre)**2
                           
                           G1 = (1 + a) * exp(- dist/(radius**2)) / (2 * pi * radius**2)
                           G2 = a * exp(- dist/(b * radius)**2) / (2 * pi * (b * radius)**2)
                           
                           w_syn = clip(G1 + G2, 0, w_syn_som_to_som_max)
                           s_lateral += w_syn
                           ''',
                           on_post='''
                           ''',
                           model='''
                           w_syn : 1 # synaptic weight / synapse efficacy
                           ''')
    som_synapse.connect(condition='i!=j')
    # som afferent connection
    temporal_to_som_synapse = Synapses(temporal_layer, target=som_layer, method=diff_method,
                                       on_pre='''
                                       s_afferent += w_syn
                                       A_pre = (- w_syn) * A_minus * (1 - 1/tau_minus) ** time_post
                                       w_syn = clip(w_syn + plasticity * A_pre, 0,  w_syn_temporal_to_som_max)
                                       ''',
                                       on_post='''
                                       A_post = exp(-w_syn) * A_plus * (1 - 1/tau_plus) * time_pre
                                       w_syn = clip(w_syn + plasticity * A_post, 0,  w_syn_temporal_to_som_max)
                                       ''',
                                       model='''
                                       w_syn : 1 # synaptic weight / synapse efficacy
                                       plasticity : boolean (shared)
                                       ''')
    temporal_to_som_synapse.connect()
    temporal_to_som_synapse.w_syn = np.random.randint(low=40000, high=60000, size=3000) \
                                    * w_syn_temporal_to_som_max / 100000.0

    # Visualization

    som_spike_mon = SpikeMonitor(som_layer)

    u_spike_mon = SpikeMonitor(temporal_layer)
    u_state_mon_v = StateMonitor(temporal_layer, 'v', record=True)
    u_state_mon_time = StateMonitor(temporal_layer, 'time', record=True)
    u_state_mon_w = StateMonitor(temporal_layer, 'w_inh2u', record=True)

    inh_spike_mon = SpikeMonitor(inhibition_neuron)
    inh_state_mon = StateMonitor(inhibition_neuron, 'v', record=True)
    w_exc_neu_state = StateMonitor(inhibition_neuron, 'w_exc', record=True)
    w_inh_neu_state = StateMonitor(inhibition_neuron, 'w_inh', record=True)

    w_syn_u2inh_exc = StateMonitor(u2inh_excitation, 'w_syn', record=True)

    defaultclock.dt = time_step * ms

    step = 2

    plasticity_state = False

    u2inh_excitation.plasticity = plasticity_state
    u2inh_inhibition.plasticity = plasticity_state
    inh2u_inhibition.plasticity = plasticity_state
    temporal_to_som_synapse.plasticity = True  # plasticity_state

    run(simulation_time * ms, report='text')

    subplot(421)
    # subplot(111)
    title("Temporal layer spikes")
    plot(u_spike_mon.t / ms, u_spike_mon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    grid(True)
    xticks(np.arange(0.0, simulation_time + step, step))
    yticks(np.arange(-1, N + 1, 1))

    # show()

    subplot(422)
    title("Inhibition neuron spikes")
    plot(inh_spike_mon.t / ms, inh_spike_mon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    grid(True)
    xticks(np.arange(0.0, simulation_time + step, step))
    yticks(np.arange(-1, 1, 1))

    subplot(423)
    title("u membrane potential")
    for item in u_state_mon_v:
        plot(u_state_mon_v.t / ms, item.v)
    # plot(u_state_mon_v.t / ms, u_state_mon_v[0].v)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    subplot(424)
    title("Inhibition neuron membrane potential")
    plot(inh_state_mon.t / ms, inh_state_mon[0].v)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    subplot(425)
    title("Excitation/inhibition interaction")
    plot(w_exc_neu_state.t / ms, w_exc_neu_state[0].w_exc, w_exc_neu_state.t / ms, w_inh_neu_state[0].w_inh,
         w_exc_neu_state.t / ms, w_exc_neu_state[0].w_exc - w_inh_neu_state[0].w_inh)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    subplot(426)
    title("Inhibition to u potential")
    plot(u_state_mon_w.t / ms, u_state_mon_w[0].w_inh2u)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    subplot(427)
    title("Synaptic Weight")
    for item in w_syn_u2inh_exc:
        plot(w_syn_u2inh_exc.t / ms, item.w_syn)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))
    yticks(np.arange(-0.1, 1.1, 0.1))

    subplot(428)
    title("Synaptic time pre spike")
    for item in u_state_mon_time:
        plot(w_syn_u2inh_exc.t / ms, item.time)
    xlabel('Time (ms)')
    ylabel('Potential')
    xticks(np.arange(0.0, simulation_time + step, step))

    show()

    # subplot(111)
    title("Som layer spikes")
    plot(som_spike_mon.t / ms, som_spike_mon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    grid(True)
    xticks(np.arange(0.0, simulation_time + step, step))
    yticks(np.arange(-1, map_size * map_size + 1, 1))

    show()
