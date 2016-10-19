using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Encog.ML.EA.Train;
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

        static void Main(string[] args)
        {
            var trainingData = CsvHelper.Read(FILENAME);
            var testData = CsvHelper.Read(TEST_FILENAME);


            var settings = new NeuralSettings
            {
                HasBias = true,
                ActivationFunction = ActivationFunction.Unipolar,
                HiddenLayers = new[] { 20, 10 },
                LearningRate = 0.003,
                Momentum = 0.3,
                Iterations = 5000,
                Type = ProblemType.Regression
            };

            var neuralWrapper = new NeuralWrapper(settings, 1, 1);
            neuralWrapper.BuildNetwork();
            neuralWrapper.Train(trainingData);


            var result = neuralWrapper.Test(testData);


            CsvHelper.Write("sortedInput.csv", SortByColumn(trainingData, 0));
            CsvHelper.Write("output.csv", testData, result);


            new GnuplotScriptRunner(@"gnuplot/regression.gnu")
                .AddScriptParameter("trainingSet", Path.GetFullPath("sortedInput.csv"))
                .AddScriptParameter("testSet", Path.GetFullPath("output.csv"))
                .Run();

        }
        public static double[][] SortByColumn(double[][] data, int column)
        {
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }
            if (column >= data[0].Length)
            {
                throw new ArgumentOutOfRangeException(nameof(column));
            }

            return data.Select(x => x).OrderByDescending(x => x[column]).ToArray();
        }
    }
}
