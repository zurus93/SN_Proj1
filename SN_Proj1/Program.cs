using System;
using System.IO;
using System.Linq;

namespace SN_Proj1
{
    class Program
    {
        private static String FILENAME = "Classification/data.simple.train.100.csv";
        private static String TEST_FILENAME = "Classification/data.simple.test.100.csv";

        private static int CLUSTERS_COUNT = 0;

        static void Main(string[] args)
        {
            var trainingData = CsvHelper.Read(FILENAME);
            var validationData = CsvHelper.Read(TEST_FILENAME);

            var settings = new NeuralSettings
            {
                HasBias = true,
                ActivationFunction = ActivationFunction.Bipolar,
                HiddenLayers = new[] { 40, 30 },
                LearningRate = 0.003,
                Momentum = 0.03,
                Iterations = 200,
                Type = ProblemType.Classification
            };

            // Zakładamy, że liczba elementów w każdej kolumnie jest taka sama i że w ostatniej kolumnie
            // znajduje się wynik
            int inputSize = trainingData[0].Length - 1;
            int idealSize = settings.Type == ProblemType.Regression ? 1 : CLUSTERS_COUNT;

            var testData = validationData.Select(x => x.Take(2).ToArray()).ToArray();
            var neuralWrapper = new NeuralWrapper(settings, inputSize, idealSize);
            neuralWrapper.BuildNetwork();
            var error = neuralWrapper.Train(trainingData, validationData);

            var result = neuralWrapper.Test(testData);

            CsvHelper.Write("sortedInput.csv", SortByColumn(trainingData, 0));
            CsvHelper.Write("output.csv", testData, result);
            CsvHelper.Write("error.csv", error);


            new GnuplotScriptRunner(@"gnuplot/regression.gnu")
                .AddScriptParameter("trainingSet", Path.GetFullPath("sortedInput.csv"))
                .AddScriptParameter("testSet", Path.GetFullPath("output.csv"))
                .Run();

            new GnuplotScriptRunner(@"gnuplot/networkError.gnu")
                .AddScriptParameter("input", Path.GetFullPath("error.csv"))
                .Run();

            Console.ReadLine();
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
