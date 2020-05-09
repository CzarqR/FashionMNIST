using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FashionMNIST
{
    class Program
    {
        static void Main()
        {
            FashionMNISTNeuralNetwork.CreateNeuralNetwork(1, 0.1, "network4l.txt");
            FashionMNISTNeuralNetwork.TestNetworkFromFile("network4l.txt");
            //FashionMNISTNeuralNetwork.Test2();

            Console.ReadKey();
        }
    }
}
