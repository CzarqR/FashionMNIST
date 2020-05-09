using System;
using System.Text;

namespace FashionMNIST.NeuralNetwork
{
    class Layer
    {
        public Neuron[] Neurons
        {
            get; set;
        }

        public Func<double, double> ActivateFun = value => 1 / (1 + Math.Pow(Math.E, -value)); //default sigmoid
        public Func<double, double> DetActivateFun = value => value * (1 - value); //default det sigmoid
        public double Bias
        {
            get; set;
        }



        public Layer(Neuron[] neurons)
        {
            Bias = 0;
            Neurons = neurons;
        }

        public Layer(int size, int nextSize)
        {
            Neurons = new Neuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(nextSize);
            }
        }

        public void ClearValues()
        {
            foreach (Neuron neuron in Neurons)
            {
                neuron.Value = 0;
            }
        }

        public void UpdateNeurons()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].OutValue = ActivateFun(Neurons[i].Value + Bias);
            }
        }

        public void UpdateSynapses()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].UpdateSynapses();
            }
        }

        public override string ToString()
        {
            StringBuilder sr = new StringBuilder();

            for (int i = 0; i < Neurons.Length; i++)
            {
                sr.Append(Math.Round(Neurons[i].OutValue, 3) + "   ");
            }
            sr.Append('\n');
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < Neurons[i].Synapses.Length; j++)
                {
                    sr.Append(Math.Round(Neurons[i].Synapses[j].Weight, 3) + "  ");
                }
                if (Neurons[i].Synapses.Length > 0)
                {
                    sr.Append(" |  ");
                }
            }
            sr.Append('\n');


            return sr.ToString();

        }
    }
}
