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

            Network network = new Network(784, 16, 16, 10)
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

            Network network = new Network(784, 16, 16, 10); // stworzenie sieci takiej samej jak w funkcji wyżej żeby można było  wczytać wagi z pliku
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





        public static void Test2()
        {
            double[,] data = LoadDataFashion("test.txt");

            //for (int i = 0; i < data.GetLength(0); i++)
            //{
            //    for (int j = 0; j < data.GetLength(1); j++)
            //    {
            //        Console.Write(data[i, j] + "\t");
            //    }
            //    Console.WriteLine("\n\n\n");
            //}

            const int LEARNING_LOOPS = 1;

            int trainingSize = data.GetLength(0);

            Network network = new Network(784, 16, 16, 10)
            {
                LearningRate = 0.1
            };

            network.SaveToFile("test2n1.txt");


            double[] input = new double[784];
            double[] target = new double[10];

            /// uczenie 
            for (int i = 0; i < LEARNING_LOOPS; i++)
            {
                for (int j = 0; j < trainingSize; j++)
                {
                    for (int k = 0; k < input.Length; k++)
                    {
                        input[k] = data[j, k];
                    }

                    for (int k = 0; k < target.Length; k++)
                    {
                        target[k] = data[j, k + 784];
                    }

                    //Console.WriteLine("\n\nINPUT:");
                    //Console.WriteLine(string.Join("| ", input));
                    //Console.WriteLine("\n\nTARGET:");
                    //Console.WriteLine(string.Join("| ", target));

                    network.BackPropagation(input, target);
                    //Console.WriteLine(network);
                    //Console.WriteLine("\n\n\n\n\n\n\n");
                    //Console.WriteLine("ERROR: " + network.TotalError(target));
                }
            }
            //Console.WriteLine(network);







            /// Zapis do pliku
            network.SaveToFile("test2n2.txt");



            int testingSize = data.GetLength(0);



            /// testowanie 
            int correct = 0;

            for (int j = 0; j < testingSize; j++)
            {
                for (int k = 0; k < input.Length; k++)
                {
                    input[k] = data[j, k];
                }

                for (int k = 0; k < target.Length; k++)
                {
                    target[k] = data[j, k + 784];
                }

                //sprawdzenie czy najwieksza wartośc na wyjściu sieci neuronowej znajduje sie na indeksie gdzie target[index] = 1. jeśli tak to wynik jest prawidłowy
                double[] result = network.GetResult(input);
                //Console.WriteLine(string.Join(" | ", result));
                //Console.WriteLine();
                //Console.WriteLine(string.Join(" | ", target));
                //Console.WriteLine();
                //Console.WriteLine(string.Join(" | ", input));
                int maxIndex = 0;
                for (int i = 1; i < result.Length; i++)
                {
                    if (result[maxIndex] < result[i])
                    {
                        maxIndex = i;
                    }
                }

                if (target[maxIndex] == 1)
                {
                    correct++;
                }

            }

            Console.WriteLine($"Poprawne: {correct}");
            Console.WriteLine($"Liczebność zbioru testowego: {testingSize}");
            Console.WriteLine($"Skuteczność sieci: {(double)correct / (testingSize)}");

        }



        public static double[,] LoadDataFashion(string fileName)
        {
            string[] lines = File.ReadAllLines(fileName);
            double[,] data = new double[lines.Length, 794];
            string[] buff;

            /// pobranie bazy do tablicy dwuwymiarowej data
            for (int i = 0; i < lines.Length; i++)
            {
                buff = lines[i].Split(',');
                for (int j = 0; j < buff.Length - 1; j++)
                {
                    data[i, j] = double.Parse(buff[j + 1]);
                }


                if (buff[0].Equals("0"))
                {
                    data[i, 784] = 1;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 0;

                }
                else if (buff[0].Equals("1"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 1;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("2"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 1;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("3"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 1;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("4"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 1;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("5"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 1;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("6"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 1;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("7"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 1;
                    data[i, 792] = 0;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("8"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 1;
                    data[i, 793] = 0;
                }
                else if (buff[0].Equals("9"))
                {
                    data[i, 784] = 0;
                    data[i, 785] = 0;
                    data[i, 786] = 0;
                    data[i, 787] = 0;
                    data[i, 788] = 0;
                    data[i, 789] = 0;
                    data[i, 790] = 0;
                    data[i, 791] = 0;
                    data[i, 792] = 0;
                    data[i, 793] = 1;
                }
            }

            for (int i = 0; i < data.GetLength(0); i++)
            {
                double max = data[i, 0], min = data[i, 0];
                for (int j = 1; j < 784; j++)
                {
                    if (data[i, j] > max)
                    {
                        max = data[i, j];
                    }
                    else if (data[i, j] < min)
                    {
                        min = data[i, j];
                    }
                }
                for (int j = 0; j < 784; j++)
                {
                    data[i, j] = (data[i, j] - min) / (max - min);
                }
            }


            return data;
        }



    }
}
