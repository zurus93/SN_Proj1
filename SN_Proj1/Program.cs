using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;using System;
using System.Collections.Generic;
using System.Linq;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Util.CSV;
using Encog.Neural.Data.Basic;
using Encog.Util.Arrayutil;
using Encog.Util;

namespace SN_Proj1
{
    class Program
    {
        private static String FILENAME = "data.xsq.train.csv";
        private static String TEST_FILENAME = "data.xsq.test.csv";
        private static int LAYERS = 4;
        private static int NEURONS = 30;
        private static int ITERATIONS = 100;
        private static double LEARNING_RATE = 0.7;
        private static double MOMENTUM = 0.3;

        private static BasicNetwork _network;

        static void Main(string[] args)
        {
            var trainingSet = readCSV(FILENAME, false);
            var testingSet = readCSV(TEST_FILENAME, true);

            try
            {
                _network = (BasicNetwork)SerializeObject.Load(FILENAME + ".ser");
            }
            catch (System.IO.FileNotFoundException)
            {
                _network = new BasicNetwork();
                initializeNetwork(_network, trainingSet.InputSize, trainingSet.IdealSize);
            }                      

            Console.WriteLine("Press any key to start");
            Console.WriteLine("Press ESC to stop");           
            Console.ReadKey();

            trainNetwork(trainingSet);
            SerializeObject.Save(FILENAME + ".ser", _network); 

            Console.WriteLine("Press any key to start testing");
            Console.ReadKey();
           
            testNetwork(testingSet);

            Console.WriteLine("Press any key to leave");
            Console.ReadKey();
        }

        private static void testNetwork(IMLDataSet testingSet)
        {
            foreach (IMLDataPair pair in testingSet)
            {
                IMLData output = _network.Compute(pair.Input);
                Console.WriteLine(@"Computet result = " + output[0]);
            }
        }

        private static void trainNetwork(IMLDataSet trainingSet)
        {
            var train = new Backpropagation(_network, trainingSet, LEARNING_RATE, MOMENTUM);

            int epoch = 1;
            do
            {
                while (!Console.KeyAvailable)
                {
                    train.Iteration();
                    Console.WriteLine(@"Epoch #" + epoch + @" Error: " + train.Error);
                    epoch++;
                }
            } while (train.Error > 0.004 & Console.ReadKey(true).Key != ConsoleKey.Escape);
            train.FinishTraining();
        }

        private static void initializeNetwork(BasicNetwork network, int inputSize, int idealSize)
        {
            network.AddLayer(new BasicLayer(inputSize));

            for (int i = 0; i < LAYERS; ++i)
            {
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, NEURONS));
            }

            network.AddLayer(new BasicLayer(idealSize));
            network.Structure.FinalizeStructure();
            network.Reset();
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

        private static IMLDataSet readCSV(string path, bool test)
        {
            var reader = new ReadCSV(path, true, CSVFormat.DecimalPoint);
            var data = LoadCSV(reader);
            data = Normalize(data);
            reader.Close();

            double[][] input;
            double[][] ideal;

            if (test)
            {
                input = data; // testing set doesn't have 'ideal' row
                ideal = data;
            }
            else
            {
                input = data.Select(row => row.Take(row.Length - 1).ToArray()).ToArray();
                ideal = data.Select(row => row.Skip(row.Length - 1).ToArray()).ToArray(); // last value
            }

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
