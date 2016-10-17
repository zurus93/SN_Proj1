using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.ML.Train;
using Encog.ML.Data.Basic;
using Encog;using System;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Util.CSV;
using Encog.Neural.Data.Basic;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Normalizers.Strategy;
using Encog.ML.Model;
using Encog.ML.Factory;
using Encog.Util.Simple;

namespace SN_Proj1
{
    class Program
    {
        private static int LAYERS = 10;
        private static int NEURONS = 30;
        private static int ITERATIONS = 100;
        private static double LEARNING_RATE = 0.7;
        private static double MOMENTUM = 0;

        static void Main(string[] args)
        {
            var trainingSet = readCSV();

            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(1));

            for (int i = 0; i < LAYERS; ++i)
            {
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, NEURONS));
            }

            network.AddLayer(new BasicLayer(1));
            network.Structure.FinalizeStructure();
            network.Reset();

            var train = new Backpropagation(network, trainingSet, LEARNING_RATE, MOMENTUM);

            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" Error: " + train.Error);
                epoch++;
            } while (train.Error > 0.01);
            train.FinishTraining();

            Console.ReadKey();
        }

        private static IMLDataSet readCSV()
        {
            var reader = new ReadCSV("data.xsq.train.csv", true, CSVFormat.DecimalPoint);

            double[][] input = new double[1000][];
            double[][] ideal = new double[1000][];

            int line = 0;

            
            while (reader.Next())
            {
                int inputSize = reader.GetCount() - 1;
                input[line] = new double[inputSize];
                for (int i = 0; i < inputSize; ++i)
                {
                    input[line][i] = reader.GetDouble(i) / 1000;                   
                }
                ideal[line] = new double[1];
                ideal[line][0] = reader.GetDouble(inputSize) / 1000;

                ++line;
            }

            var dataSet = new BasicNeuralDataSet(input, ideal);
            reader.Close();
            return dataSet;
/*
            var source = new CSVDataSource("data.xsq.train.csv", true, CSVFormat.DecimalPoint);
            var data = new VersatileMLDataSet(source);
            

            data.DefineSourceColumn("x", ColumnType.Continuous);
            ColumnDefinition output = data.DefineSourceColumn("y", ColumnType.Nominal);          

            data.Analyze();

            data.DefineSingleOutputOthersInput(output);
            EncogModel model = new EncogModel(data);
            model.SelectMethod(data, MLMethodFactory.TypeFeedforward);

            data.Normalize();

            return data;
            */          
        }
    }
}
