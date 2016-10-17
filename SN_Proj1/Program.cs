using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.ML.Train;
using Encog.ML.Data.Basic;
using Encog;
namespace SN_Proj1
{
    class Program
    {
        static void Main(string[] args)
        {

            new GnuplotScriptRunner(GnuplotScriptRunner.RegressionScriptPath)
                .AddScriptParameter("trainingSet",
                    @"C:\Users\wardzinskim\Downloads\Regression\data.square.test.1000.csv")
                .AddScriptParameter("testSet", @"C:\Users\wardzinskim\Downloads\Regression\data.square.test.100.csv")
                .Run();
        }
    }
}
