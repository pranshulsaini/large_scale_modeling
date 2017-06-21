import pyNN.nest as sim
import numpy as np
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.parameters import Sequence
from pyNN.random import RandomDistribution as rnd
import matplotlib.pyplot as plt
plt.interactive(False)
sim.setup(0.1)

neurons = sim.Population(64, sim.IF_curr_exp,{}, label="neurons")
print(neurons.celltype.recordable)

params = {
        'rate':     10000.0,  # Mean spike frequency (Hz)
        'start':    0.0,  # Start time (ms)
        'duration': 1e10  # Duration of spike sequence (ms)
    }

input = sim.Population(64, sim.SpikeSourcePoisson(**params), label="input")

input_proj = sim.Projection(input, neurons, sim.OneToOneConnector(), receptor_type= 'excitatory') ##https://github.com/NeuralEnsemble/PyNN/issues/273

def twoDto1D(x):
    return (8*x[0] + x[1])

def oneDto2D(n):
    return np.array([int(n/8.), int(n -8*(int(n/8.)))])

def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)


#excitatory connections
for i in range(63):
    sim.connect(neurons[i], neurons[i], receptor_type= 'excitatory')

    [x,y] = oneDto2D(i)

    m = x
    n = y
    while(m>0 & n>0):
        m = m-1
        n = n-1
        j = twoDto1D([m,n])
        sim.connect(neurons[i], neurons[j], receptor_type='inhibitory')

    m = x
    n = y
    while(m>0 & n<7):
        m = m-1
        n = n+1
        j = twoDto1D([m,n])
        sim.connect(neurons[i], neurons[j], receptor_type='inhibitory')

    m = x
    n = y
    while (m > 0):
        m = m - 1
        j = twoDto1D([m, n])
        sim.connect(neurons[i], neurons[j],  receptor_type='inhibitory')

    m = x
    n = y
    while (m <7 ):
        m = m + 1
        j = twoDto1D([m, n])
        sim.connect(neurons[i], neurons[j], receptor_type='inhibitory')

    m = x
    n = y
    while (m < 7 & n > 0):
        m = m + 1
        n = n - 1
        j = twoDto1D([m, n])
        sim.connect(neurons[i], neurons[j], receptor_type='inhibitory')

    m = x
    n = y
    while (m < 7 & n < 7):
        m = m + 1
        n = n + 1
        j = twoDto1D([m, n])
        sim.connect(neurons[i], neurons[j], receptor_type='inhibitory')


neurons.record('spikes')
sim.run(100)  # in ms

data = neurons.get_data()
print(data)


plot_spiketrains(data.segments[0])
plt.show()


    #input = sim.Population(1, sim.SpikeSourceArray,{'spike_times': [[0]]}, label="input")

