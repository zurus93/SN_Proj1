using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Encog.Engine.Network.Activation;
using Encog.Neural.Networks.Training.Simple;

namespace SN_Proj1
{
    class Program
    {
        private static String FILENAME = "Classification/data.three_gauss.train.10000.csv";
        private static String TEST_FILENAME = "Classification/data.three_gauss.test.10000.csv";

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
            if (validationData[0].Length == inputSize)
            {
                validationData = null;
            }


            var neuralWrapper = new NeuralWrapper(settings, inputSize, idealSize);
            neuralWrapper.BuildNetwork();
            var error = neuralWrapper.Train(trainingData, validationData);

            var result = neuralWrapper.Test(testData);



            CsvHelper.Write("error.csv", error);
            CsvHelper.Write("sortedInput.csv", SortByColumn(trainingData, 0));
            CsvHelper.Write("output.csv", testData, result);

            if (settings.Type == ProblemType.Regression)
            {
                new GnuplotScriptRunner(@"gnuplot/regression.gnu")
                    .AddScriptParameter("trainingSet", Path.GetFullPath("sortedInput.csv"))
                    .AddScriptParameter("testSet", Path.GetFullPath("output.csv"))
                    .Run();
            }
            else
            {
                new GnuplotScriptRunner(@"gnuplot/classification.gnu")
                    .AddScriptParameter("trainingSet", Path.GetFullPath("sortedInput.csv"))
                    .AddScriptParameter("testSet", Path.GetFullPath("output.csv"))
                    .Run();
            }


            new GnuplotScriptRunner(@"gnuplot/networkError.gnu")
                .AddScriptParameter("input", Path.GetFullPath("error.csv"))
                .Run();

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
                new CommandLineOption {ShortNotation = 'o', LongNotation = "test-data", ParameterRequired = true}
            });

            parser.Parse(arguments);
            var settings = new NeuralSettings { HiddenLayers = new int[0], HasBias = true };

            string value = null;
            if (parser.TryGet("iterations", out value))
                settings.Iterations = int.Parse(value);
            else
                settings.Iterations = 200;

            if (parser.TryGet("momentum", out value))
                settings.Momentum = double.Parse(value);
            else
                settings.Momentum = 0.3;

            if (parser.TryGet("learning-rate", out value))
                settings.LearningRate = double.Parse(value);
            else
                settings.LearningRate = 0.07;

            if (parser.TryGet("bias", out value))
                settings.HasBias = bool.Parse(value);
            else
                settings.HasBias = true;

            if (parser.TryGet("activation-function", out value))
            {
                ActivationFunction tmp;
                Enum.TryParse(value, true, out tmp);
                settings.ActivationFunction = tmp;
            }
            else
                settings.ActivationFunction = ActivationFunction.Unipolar;

            if (parser.TryGet("problem-type", out value))
            {
                ProblemType tmp;
                Enum.TryParse(value, true, out tmp);
                settings.Type = tmp;
            }
            else
                settings.Type = ProblemType.Classification;

            if (parser.TryGet("hidden-layers", out value))
                settings.HiddenLayers = value.Split(',').Select(int.Parse).ToArray();
            else
                settings.HiddenLayers = new int[] { 40, 30 };

            if (parser.TryGet("training-data", out value))
                FILENAME = value;

            if (parser.TryGet("test-data", out value))
                TEST_FILENAME = value;

            return settings;
        }
    }
}
