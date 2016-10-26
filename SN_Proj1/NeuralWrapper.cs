using System;
using System.Collections.Generic;
using System.Linq;
using Encog.Engine.Network.Activation;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.ML.Data;

namespace SN_Proj1
{
    public class NeuralWrapper
    {
        private readonly NeuralSettings _settings;
        private BasicNetwork _network;
        private readonly int _inputLayerSize, _outputLayerSize;
        private DataNormalizer _normalizer;

        public NeuralWrapper(NeuralSettings settings, int inputSize, int outputSize)
        {
            if (settings == null)
            {
                throw new ArgumentNullException(nameof(settings));
            }

            _settings = settings;
            _inputLayerSize = inputSize;
            _outputLayerSize = outputSize;
        }


        public void BuildNetwork()
        {
            _network = new BasicNetwork();
            _network.AddLayer(new BasicLayer(null, _settings.HasBias, _inputLayerSize));

            if (_settings.HiddenLayers != null)
            {
                foreach (var hiddenLayerSize in _settings.HiddenLayers)
                {
                    _network.AddLayer(new BasicLayer(GetActivationFunction(), _settings.HasBias, hiddenLayerSize));
                }
            }

            _network.AddLayer(new BasicLayer(GetActivationFunction(), _settings.HasBias, _outputLayerSize));
            _network.Structure.FinalizeStructure();
            _network.Reset();

        }

        public IActivationFunction GetActivationFunction()
        {
            switch (_settings.ActivationFunction)
            {
                case ActivationFunction.Unipolar:
                    return new ActivationSigmoid();
                case ActivationFunction.Bipolar:
                    return new ActivationTANH();
            }
            return null;
        }

        public double[][] Train(double[][] trainingData, double[][] validationData)
        {
            var error = new List<double[]>();
            PrepareNormalizerFor(trainingData, validationData);

            var trainingSet = PrepareSet(trainingData);
            var validationSet = PrepareSet(validationData);

            var training = new Backpropagation(_network, trainingSet, _settings.LearningRate, _settings.Momentum)
            {
                BatchSize = 1
            };

            for (int epoch = 0; epoch < _settings.Iterations; epoch++)
            {
                training.Iteration();
                double trainingError = -1;
                double testingError = -1;

                if (_settings.Type == ProblemType.Regression)
                {
                    trainingError = _network.CalculateError(trainingSet);
                    if (validationSet != null)
                    {
                        testingError = _network.CalculateError(validationSet);
                    }
                }
                else
                {
                    trainingError = CalculateClassificationError(trainingSet);
                    if (validationSet != null)
                    {
                        testingError = CalculateClassificationError(validationSet);
                    }
                }
                var errorIter = new[] { epoch, trainingError, testingError };
                error.Add(errorIter);
                Console.WriteLine($"Epoch #{epoch} [{training.Error}] TrainingError: {errorIter[1]} ValidationError: {errorIter[2]}");
            }
            training.FinishTraining();

            return error.ToArray();
        }

        private double CalculateClassificationError(BasicNeuralDataSet trainingSet)
        {
            int errorCount = 0;
            foreach (var trainData in trainingSet)
            {
                IMLData output = _network.Compute(trainData.Input);
                IMLData ideal = trainData.Ideal;

                double maxValue = Double.MinValue;
                int maxIndex = 0;

                for (int i = 0; i < output.Count; ++i)
                {
                    if (maxValue < output[i])
                    {
                        maxValue = output[i];
                        maxIndex = i;
                    }
                }

                if (Math.Abs(ideal[maxIndex] - 1) > 0.0001)
                    errorCount++;
            }

            return (double)errorCount / trainingSet.Count;
        }

        private double GetNormalizationLowValue()
        {
            if (_settings.ActivationFunction == ActivationFunction.Bipolar)
            {
                return -1;
            }
            return 0;
        }

        private void PrepareNormalizerFor(double[][] trainingData, double[][] validationData)
        {
            List<double[]> dataToNormalize = new List<double[]>();

            if (trainingData != null)
            {
                dataToNormalize.AddRange(trainingData);
            }
            if (validationData != null)
            {
                dataToNormalize.AddRange(validationData);
            }

            _normalizer = new DataNormalizer(dataToNormalize.ToArray(), 1, _settings.ActivationFunction == ActivationFunction.Unipolar ? 0 : -1);
        }

        private Tuple<double[][], double[][]> Split(double[][] data, double percent)
        {
            var rnd = new Random();

            var columns = Enumerable.Range(0, data.Length)
                .Select(x => new
                {
                    Index = rnd.Next(),
                    Value = x
                }).OrderBy(x => x.Index).Take((int)(data.Length * percent)).Select(x => x.Value).ToArray();

            var learningSet = data.Where((row, index) => !columns.Contains(index)).ToArray();
            var validationSet = data.Where((row, index) => columns.Contains(index)).ToArray();
            return new Tuple<double[][], double[][]>(learningSet, validationSet);
        }


        public double[][] Test(double[][] testData)
        {
            var normalizedData = _normalizer.Normalize(testData, Enumerable.Range(0, _inputLayerSize).ToArray());
            var result = new List<double[]>();
            Console.WriteLine();
            foreach (var input in normalizedData)
            {
                var output = new double[_outputLayerSize];
                _network.Compute(input, output);

                if (_settings.Type == ProblemType.Regression)
                {
                    result.Add(_normalizer.Denormalize(output,
                        Enumerable.Range(_inputLayerSize, output.Length).ToArray())); // Denormalize if needed
                }
                else
                {
                    result.Add(new double[] { output.ToList().IndexOf(output.Max()) + 1 });
                }
            }
            return result.ToArray();
        }


        private BasicNeuralDataSet PrepareSet(double[][] data)
        {
            if (data == null)
                return null;
            return _settings.Type == ProblemType.Regression ? PrepareRegressionSet(data) : PrepareClassificationSet(data);
        }

        private BasicNeuralDataSet PrepareRegressionSet(double[][] data)
        {
            var normalizedData = _normalizer.Normalize(data);

            var input = normalizedData.Select(row => row.Take(_inputLayerSize).ToArray()).ToArray();
            var ideal = normalizedData.Select(row => row.Skip(_inputLayerSize).ToArray()).ToArray();

            var trainingSet = new BasicNeuralDataSet(input, ideal);

            return trainingSet;
        }

        private BasicNeuralDataSet PrepareClassificationSet(double[][] data)
        {
            var input = data.Select(row => row.Take(_inputLayerSize).ToArray()).ToArray();
            var idealTmp = data.Select(row => row.Skip(_inputLayerSize).ToArray()).ToArray();

            var normalizedInput = _normalizer.Normalize(input);
            var ideal = new double[idealTmp.Length][];

            // Dzielimy wynik na klasy, "uaktywniając" odpowiadającą klasie kolumnę
            for (int i = 0; i < idealTmp.Length; ++i)
            {
                ideal[i] = Enumerable.Repeat(GetNormalizationLowValue(), _outputLayerSize).ToArray();

                ideal[i][(int)idealTmp[i][0] - 1] = 1;
            }

            var trainingSet = new BasicNeuralDataSet(normalizedInput, ideal);

            return trainingSet;
        }
    }
}
