using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Encog.Neural.Networks.Training.Simple;

namespace SN_Proj1
{
    class Program
    {
        private static String FILENAME = "Classification/data.simple.train.100.csv";
        private static String TEST_FILENAME = "Classification/data.simple.test.100.csv";

        private static int CLUSTERS_COUNT = 3;

        static void Main(string[] args)
        {
            var settings = ApplyArguments(args);

            var trainingData = CsvHelper.Read(FILENAME);
            var validationData = CsvHelper.Read(TEST_FILENAME);


            // Zakładamy, że liczba elementów w każdej kolumnie jest taka sama i że w ostatniej kolumnie
            // znajduje się wynik
            int inputSize = trainingData[0].Length - 1;
            int idealSize = settings.Type == ProblemType.Regression ? 1 : GetClastersCount(trainingData);

            var testData = validationData.Select(x => x.Take(inputSize).ToArray()).ToArray();



            var neuralWrapper = new NeuralWrapper(settings, inputSize, idealSize);
            neuralWrapper.BuildNetwork();
            var error = neuralWrapper.Train(trainingData, validationData);

            var result = neuralWrapper.Test(testData);

            if (settings.Type == ProblemType.Regression)
            {

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

            }
            Console.ReadLine();
        }

        public static int GetClastersCount(double[][] data)
        {
            return (int)data.Select(x => x.Last()).Max();
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

        public static NeuralSettings ApplyArguments(string[] arguments)
        {
            var parser = new CommandLineParser(new List<CommandLineOption>
            {
                new CommandLineOption {ShortNotation = 'i', LongNotation = "iterations", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 'h', LongNotation = "hidden-layers", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 't', LongNotation = "problem-type", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 'f', LongNotation = "activation-function", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 'b', LongNotation = "bias", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 'l', LongNotation = "learning-rate", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 'm', LongNotation = "momentum", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 'd', LongNotation = "training-data", ParameterRequired = true},
                new CommandLineOption {ShortNotation = 'v', LongNotation = "test-data", ParameterRequired = true}
            });

            parser.Parse(arguments);
            var settings = new NeuralSettings { HiddenLayers = new int[0], HasBias = true };

            string value = null;
            if (parser.TryGet("iterations", out value))
                settings.Iterations = int.Parse(value);

            if (parser.TryGet("momentum", out value))
                settings.Momentum = double.Parse(value);

            if (parser.TryGet("learning-rate", out value))
                settings.LearningRate = double.Parse(value);

            if (parser.TryGet("bias", out value))
                settings.HasBias = bool.Parse(value);

            if (parser.TryGet("activation-function", out value))
            {
                ActivationFunction tmp;
                Enum.TryParse(value, true, out tmp);
                settings.ActivationFunction = tmp;
            }

            if (parser.TryGet("problem-type", out value))
            {
                ProblemType tmp;
                Enum.TryParse(value, true, out tmp);
                settings.Type = tmp;
            }

            if (parser.TryGet("hidden-layers", out value))
                settings.HiddenLayers = value.Split(',').Select(int.Parse).ToArray();

            if (parser.TryGet("training-data", out value))
                FILENAME = value;

            if (parser.TryGet("test-data", out value))
                TEST_FILENAME = value;

            return settings;
        }
    }
}
