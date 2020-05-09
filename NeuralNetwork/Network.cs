using System;
using System.IO;
using System.Text;

namespace FashionMNIST.NeuralNetwork
{
    class Network
    {
        public Layer[] Layers
        {
            get; private set;
        }

        public double LearningRate
        {
            get; set;
        }

        public Network(Layer[] layers)
        {
            LearningRate = 0.05;
            Layers = layers;
        }

        //tworzy sieć z losowymi wagami pomiędzy 0 a 1
        public Network(params int[] sizes)
        {
            LearningRate = 0.5;
            Layers = new Layer[sizes.Length];
            for (int i = 0; i < Layers.Length - 1; i++)
            {
                Layers[i] = new Layer(sizes[i], sizes[i + 1]);
            }
            Layers[Layers.Length - 1] = new Layer(sizes[Layers.Length - 1], 0);

            Layers[0].ActivateFun = value => value; //domyślnie wartwa input ma funkcje aktywacji liniową
        }



        public void Calculate(double[] input)
        {
            for (int i = 0; i < input.Length; i++)
            {
                Layers[0].Neurons[i].Value = input[i];
            }

            for (int i = 0; i < Layers.Length - 1; i++)
            {
                Layers[i + 1].ClearValues();
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    for (int k = 0; k < Layers[i + 1].Neurons.Length; k++)
                    {
                        Layers[i + 1].Neurons[k].Value += ((Layers[i].ActivateFun(Layers[i].Neurons[j].Value + Layers[i].Bias)) * Layers[i].Neurons[j].Synapses[k].Weight);
                    }
                }
            }

        }

        public double[] GetResult(double[] input)
        {
            Calculate(input);
            UpdateNeutronsValues();
            double[] result = new double[Layers[Layers.Length - 1].Neurons.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Layers[Layers.Length - 1].Neurons[i].OutValue;
            }
            return result;
        }

        public void UpdateNeutronsValues()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].UpdateNeurons();
            }
        }

        void UpdateSynapesValues()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].UpdateSynapses();
            }
        }


        double[] CalculateError(double[] target)
        {
            double[] r = new double[target.Length];
            for (int i = 0; i < target.Length; i++)
            {
                r[i] = (target[i] - Layers[Layers.Length - 1].Neurons[i].OutValue) * (target[i] - Layers[Layers.Length - 1].Neurons[i].OutValue) / 2;
            }
            return r;
        }

        public double TotalError(double[] target)
        {
            double[] errors = CalculateError(target);
            double s = 0;
            foreach (double error in errors)
            {
                s += Math.Abs(error);
            }
            return s;
        }

        void BackPropLastLayer(double[] target)
        {
            for (int i = 0; i < Layers[Layers.Length - 2].Neurons.Length; i++) //iteruje po wszystkich neuronach ostatniej wartswy hidden
            {
                for (int j = 0; j < Layers[Layers.Length - 2].Neurons[i].Synapses.Length; j++)//iteruje po kazdej synapsie danego neuronu w warstwie
                {
                    double det_error = (Layers[Layers.Length - 1].Neurons[j].OutValue - target[j]) * (Layers[Layers.Length - 1].DetActivateFun(Layers[Layers.Length - 1].Neurons[j].OutValue));
                    Layers[Layers.Length - 1].Neurons[j].Det_Error = det_error; //wartość używana w następnej iteracji, nie będzie trzeba liczyć. (to jest CHYBA (pochodna błedu po wartości wyjściowej) * (pochodna z funkcji aktywacji) )
                    Layers[Layers.Length - 2].Neurons[i].Synapses[j].NewWeight -= ((Layers[Layers.Length - 2].Neurons[i].OutValue * det_error) * LearningRate);
                }
            }
        }

        void BackPropMiddleLayer(int m)
        {
            for (int k = 0; k < Layers[m].Neurons.Length; k++) //iteracja po kazdym neuronie w warstwie
            {
                for (int j = 0; j < Layers[m].Neurons[k].Synapses.Length; j++) //iteracja po kazdej synapsie neuronu z wartwy m
                {
                    double s = 0;
                    for (int i = 0; i < Layers[m + 1].Neurons[0].Synapses.Length; i++) //iteracja po kazdej synapsie z neuronu z wartwy m+1
                    {
                        s += (Layers[m + 2].Neurons[i].Det_Error * Layers[m + 1].Neurons[j].Synapses[i].Weight);
                    }
                    double det_error = s * Layers[m + 1].DetActivateFun(Layers[m + 1].Neurons[j].OutValue); //wartość używana w następnej iteracji
                    Layers[m + 1].Neurons[j].Det_Error = det_error;
                    Layers[m].Neurons[k].Synapses[j].NewWeight -= ((det_error * Layers[m].Neurons[k].OutValue) * LearningRate);
                }
            }

        }

        void BackPropCombined(double[] target)
        {
            BackPropLastLayer(target);
            for (int i = Layers.Length - 3; i >= 0; i--)
            {
                BackPropMiddleLayer(i);
            }
        }


        public void BackPropagation(double[] input, double[] target)
        {
            Calculate(input);
            UpdateNeutronsValues();
            BackPropCombined(target);
            UpdateSynapesValues();
        }

        public void SaveToFile(string fileName)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    for (int k = 0; k < Layers[i].Neurons[j].Synapses.Length; k++)
                    {
                        sb.Append(Layers[i].Neurons[j].Synapses[k].Weight + "\n");
                    }
                }
            }
            File.WriteAllText(fileName, sb.ToString());
        }

        public void ReadFromFile(string fileName)
        {
            string[] lines = File.ReadAllLines(fileName);
            int p = 0;
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    for (int k = 0; k < Layers[i].Neurons[j].Synapses.Length; k++)
                    {
                        Layers[i].Neurons[j].Synapses[k].Weight = double.Parse(lines[p++]);
                    }
                }
            }
        }


        public override string ToString()
        {
            StringBuilder sr = new StringBuilder();

            for (int i = 0; i < Layers.Length; i++)
            {
                sr.Append(Layers[i].ToString());
            }

            return sr.ToString();
        }

    }
}
