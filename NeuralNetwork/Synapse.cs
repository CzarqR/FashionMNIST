namespace FashionMNIST.NeuralNetwork
{
    class Synapse
    {
        public double Weight
        {
            get; set;
        }
        public double NewWeight
        {
            get; set;
        }

        public Synapse(double weight = 1)
        {
            Weight = weight;
            NewWeight = Weight;
        }

        public void Update()
        {
            Weight = NewWeight;
        }
    }
}
