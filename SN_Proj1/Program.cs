using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.ML.Train;
using Encog.ML.Data.Basic;
using Encog;using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Util.CSV;
using Encog.Neural.Data.Basic;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Normalizers.Strategy;
using Encog.ML.Model;
using Encog.ML.Factory;
using Encog.Util.Arrayutil;
using Encog.Util.Normalize.Output.Nominal;
using Encog.Util.Simple;

namespace SN_Proj1
{
    class Program
    {
        private static int LAYERS = 4;
        private static int NEURONS = 30;
        private static int ITERATIONS = 100;
        private static double LEARNING_RATE = 0.2;
        private static double MOMENTUM = 0.2;

        static void Main(string[] args)
        {
            var trainingSet = readCSV("data.xsq.train.csv");

            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(1));

            for (int i = 0; i < LAYERS; ++i)
            {
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, NEURONS));
            }

            network.AddLayer(new BasicLayer(1));
            network.Structure.FinalizeStructure();
            network.Reset();

            var train = new Backpropagation(network, trainingSet);

            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" Error: " + train.Error);
                epoch++;
            } while (train.Error > 0.004);
            train.FinishTraining();

            Console.ReadKey();
        }





        private static double[] ReadLine(ReadCSV csv)
        {
            var line = new double[csv.GetCount()];
            for (int i = 0; i < csv.GetCount(); i++)
            {
                line[i] = csv.GetDouble(i);
            }
            return line;
        }


        private static double[][] LoadCSV(ReadCSV csvReader)
        {
            if (csvReader == null)
            {
                throw new ArgumentNullException(nameof(csvReader));
            }

            IList<double[]> data = new List<double[]>();

            while (csvReader.Next())
            {
                data.Add(ReadLine(csvReader));
            }
            return data.ToArray();
        }

        private static IMLDataSet readCSV(string path)
        {
            var reader = new ReadCSV(path, true, CSVFormat.DecimalPoint);
            var data = LoadCSV(reader);
            data = Normalize(data);
            reader.Close();

            var input = data.Select(row => row.Take(row.Length - 1).ToArray()).ToArray();
            var ideal = data.Select(row => row.Skip(row.Length - 1).ToArray()).ToArray(); // last value

            return new BasicNeuralDataSet(input, ideal);
        }


        private static double[][] Normalize(double[][] data)
        {
            //TODO: trzeba napisać własny normalizator tak, żebu dało się odwrócić jego dziłanie
            //https://s3.amazonaws.com/heatonresearch-books/free/Encog3CS-User.pdf strona 22

            var rows = data.Length;
            var colums = data[0].Length;

            var tmp = data.SelectMany(x => x.ToArray()).ToArray();

            var norm = new NormalizeArray { NormalizedHigh = 1, NormalizedLow = -1 };
            var afterNormalize = norm.Process(tmp);

            double[][] result = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                result[i] = new double[colums];

                for (int j = 0; j < colums; j++)
                {
                    result[i][j] = afterNormalize[i * colums + j];
                }
            }


            return result;

        }
    }
}
