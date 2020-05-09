using System;

namespace FashionMNIST.NeuralNetwork
{
    class Neuron
    {
        public double Value
        {
            get; set;
        }

        public double OutValue
        {
            get; set;
        }

        public Synapse[] Synapses
        {
            get; set;
        }

        public double Det_Error
        {
            get; set;
        }

        private static readonly Random random = new Random();


        public Neuron(Synapse[] synapses)
        {
            Synapses = synapses;
        }

        public Neuron(params double[] synapses)
        {
            Synapses = new Synapse[synapses.Length];
            for (int i = 0; i < Synapses.Length; i++)
            {
                Synapses[i] = new Synapse(synapses[i]);
            }
        }

        public Neuron(int nextSize)
        {
            Synapses = new Synapse[nextSize];
            for (int i = 0; i < Synapses.Length; i++)
            {
                Synapses[i] = new Synapse(random.NextDouble() * 2 - 1);
            }
        }

        public void UpdateSynapses()
        {
            foreach (Synapse synapse in Synapses)
            {
                synapse.Update();
            }
        }




    }
}
