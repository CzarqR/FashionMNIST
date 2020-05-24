using FashionMNIST.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FashionMNIST
{
    static class FashionMNISTNeuralNetwork
    {
        public static void CreateNeuralNetwork(int learningLoops, double learningRate, string fileName)
        {

            LoadData.LoadDataFashion("test.txt", out List<double[]> data, out List<double[]> labels);

            int trainingSize = data.Count(); // zbiór treningowy (60k wierszy)

            Network network = new Network(784, 24, 16, 10)
            {
                LearningRate = learningRate
            }; // ile neuronow ma być na warstwie, 4 wartswy


            //uczenie
            Console.WriteLine("start");
            for (int i = 0; i < learningLoops; i++)
            {
                Console.WriteLine(i);
                for (int j = 0; j < trainingSize; j++)
                {
                    network.BackPropagation(data[j], labels[j]);
                }
            }
            Console.WriteLine("end");

            network.SaveToFile(fileName);

        }


        public static void TestNetworkFromFile(string fileName)
        {
            LoadData.LoadDataFashion("test.txt", out List<double[]> data, out List<double[]> labels);

            Network network = new Network(784, 24, 16, 10);
            network.ReadFromFile(fileName);
            int correct = 0, testingSize = data.Count;



            for (int j = 0; j < testingSize; j++)
            {
                double[] result = network.GetResult(data[j]);

                int maxIndex = 0;
                for (int i = 1; i < result.Length; i++)
                {
                    if (result[maxIndex] < result[i])
                    {
                        maxIndex = i;
                    }
                }

                if (labels[j][maxIndex] == 1)
                {
                    correct++;
                }

            }

            Console.WriteLine($"Poprawne: {correct}");
            Console.WriteLine($"Liczebność zbioru testowego: {testingSize}");
            Console.WriteLine($"Skuteczność sieci: {(double)correct / (testingSize)}");
        }


        public static void CreateAndTestNeuralNetwork(int learningLoops, double learningRate, string fileName)
        {

            LoadData.LoadDataFashion("test.txt", out List<double[]> data, out List<double[]> labels);
            LoadData.LoadDataFashion("test.txt", out List<double[]> dataTest, out List<double[]> labelsTest);

            int trainingSize = data.Count();

            Network network = new Network(784, 24, 16, 16, 10)
            {
                LearningRate = learningRate
            };

            {
                int correct = 0, testingSize = dataTest.Count;
                for (int j = 0; j < testingSize; j++)
                {
                    double[] result = network.GetResult(data[j]);

                    int maxIndex = 0;
                    for (int k = 1; k < result.Length; k++)
                    {
                        if (result[maxIndex] < result[k])
                        {
                            maxIndex = k;
                        }
                    }

                    if (labelsTest[j][maxIndex] == 1)
                    {
                        correct++;
                    }
                }
                Console.WriteLine($"{0} {correct}");
            }


            //uczenie
            for (int i = 0; i < learningLoops; i++)
            {
                for (int j = 0; j < trainingSize; j++)
                {
                    network.BackPropagation(data[j], labels[j]);
                }

                int correct = 0, testingSize = dataTest.Count;
                for (int j = 0; j < testingSize; j++)
                {
                    double[] result = network.GetResult(data[j]);

                    int maxIndex = 0;
                    for (int k = 1; k < result.Length; k++)
                    {
                        if (result[maxIndex] < result[k])
                        {
                            maxIndex = k;
                        }
                    }

                    if (labelsTest[j][maxIndex] == 1)
                    {
                        correct++;
                    }
                }
                Console.WriteLine($"{i + 1} {correct}");
                network.SaveToFile((i + 1) + fileName);
            }
            Console.WriteLine("end");

        }

    }
}
