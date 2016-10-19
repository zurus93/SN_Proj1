using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SN_Proj1
{
    public class NeuralSettings
    {
        public int[] HiddenLayers { get; set; }

        public ProblemType Type { get; set; }

        public ActivationFunction ActivationFunction { get; set; }

        public bool HasBias { get; set; }

        public double LearningRate { get; set; }

        public double Momentum { get; set; }

        public int Iterations { get; set; }
    }

    public enum ActivationFunction
    {
        Unipolar,
        Bipolar,
    }

    public enum ProblemType
    {
        Classification,
        Regression,
    }
}
