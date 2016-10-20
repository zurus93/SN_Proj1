using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;
using System.Xml.XPath;
using Encog.Engine.Network.Activation;
using Encog.MathUtil.Error;
using Encog.ML.Data;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Simple;

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
                    return null;
            }
            return null;
        }

        public double[][] Train(double[][] trainingData, double[][] validationData)
        {
            var error = new List<double[]>();

            var trainingSet = PrepareSet(trainingData);
            var validationSet = PrepareSet(validationData);

            var training = new Backpropagation(_network, trainingSet, _settings.LearningRate, _settings.Momentum);

            for (int epoch = 0; epoch < _settings.Iterations; epoch++)
            {
                training.Iteration();
                var errorIter = new[] { epoch, _network.CalculateError(trainingSet), _network.CalculateError(validationSet) };
                error.Add(errorIter);
                Console.WriteLine($"Epoch #{epoch} TrainingError: {errorIter[1]} ValidationError: {errorIter[2]}");
            }
            training.FinishTraining();

            return error.ToArray();
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
            foreach (var input in normalizedData)
            {
                var output = new double[_outputLayerSize];
                _network.Compute(input, output);
                result.Add(_normalizer.Denormalize(output, Enumerable.Range(_inputLayerSize, output.Length).ToArray())); // Denormalize if needed
            }
            return result.ToArray();
        }


        private BasicNeuralDataSet PrepareSet(double[][] data)
        {
            _normalizer = new DataNormalizer(data, 1, _settings.ActivationFunction == ActivationFunction.Unipolar ? 0 : -1);

            // w klasyfikacji nie trzeba normalizować: dane wzorcowe w formacie: 0,1,0 - element należy do klasy 2
            var normalizedData = _normalizer.Normalize(data);

            var input = normalizedData.Select(row => row.Take(_inputLayerSize).ToArray()).ToArray();
            var ideal = normalizedData.Select(row => row.Skip(_inputLayerSize).ToArray()).ToArray();

            var trainingSet = new BasicNeuralDataSet(input, ideal);
            return trainingSet;
        }
    }
}
