using System.Collections.Generic;
using System.IO;


namespace FashionMNIST
{
    static class LoadData
    {
        public static void LoadDataFashion(string fileName, out List<double[]> data, out List<double[]> labels)
        {
            string[] lines = File.ReadAllLines(fileName);
            data = new List<double[]>();
            labels = new List<double[]>();
            string[] buff;

            for (int i = 0; i < lines.Length; i++)
            {
                buff = lines[i].Split(',');
                double[] lb = new double[10];
                double[] db = new double[784];

                for (int j = 0; j < 10; j++)
                {
                    if (buff[0] == (j.ToString()))
                    {
                        lb[j] = 1;
                    }
                    else
                    {
                        lb[j] = 0;
                    }
                }

                for (int j = 1; j < buff.Length; j++)
                {
                    db[j - 1] = double.Parse(buff[j]);
                }

                labels.Add(lb);
                data.Add(db);
            } 

            /// normalizacja miejsc od 0 do 783 w każdym wierszu
            for (int i = 0; i < data.Count; i++)
            {
                /// szukanie w każdym wierszu wartości min i max
                double max = data[i][0], min = data[i][0];
                for (int j = 1; j < 784; j++)
                {
                    if (data[i][j] > max)
                    {
                        max = data[i][j];
                    }
                    else if (data[i][j] < min)
                    {
                        min = data[i][j];
                    }
                }

                /// zastosowanie wzoru na normalizację 
                for (int j = 0; j < 784; j++)
                {
                    data[i][j] = (data[i][j] - min) / (max - min);
                }
            }
        }
    }
}
