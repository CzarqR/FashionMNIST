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
            //FashionMNISTNeuralNetwork.CreateNeuralNetwork(50, 0.1, "network4Bloop50.txt");
            //FashionMNISTNeuralNetwork.TestNetworkFromFile("network4Bloop50.txt");
            FashionMNISTNeuralNetwork.CreateAndTestNeuralNetwork(40, 0.1, "_loops_network_784_24_16_16_10.txt");

            Console.ReadLine();
            Console.WriteLine("Sure?");
            Console.ReadLine();
        }
    }
}
