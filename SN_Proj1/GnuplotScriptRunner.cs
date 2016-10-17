using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Text;

namespace SN_Proj1
{
    public class GnuplotScriptRunner
    {
        public const string RegressionScriptPath = "gnuplot/regression.gnu";


        private readonly IDictionary<string, string> _scriptParameters;
        private readonly string _scriptPath;

        public GnuplotScriptRunner(string scriptPath)
        {
            _scriptPath = scriptPath;
            if (!File.Exists(scriptPath))
            {
                throw new FileNotFoundException("script file does not exist");
            }

            _scriptParameters = new Dictionary<string, string>();
        }

        public GnuplotScriptRunner AddScriptParameter(string key, string value)
        {
            _scriptParameters.Add(key, value);
            return this;
        }

        public void Run()
        {
            try
            {
                var gnuplotProcess = new Process
                {
                    StartInfo =
                    {
                        FileName = "gnuplot.exe",
                        WindowStyle = ProcessWindowStyle.Hidden,
                        CreateNoWindow = true,
                        RedirectStandardInput = true,
                        UseShellExecute = false,
                        RedirectStandardOutput = false,
                        Arguments = $"{GetScriptParameters()} {_scriptPath}"
                    }
                };

                gnuplotProcess.Start();
                gnuplotProcess.WaitForExit();
            }
            catch (Exception ex)
            {
                //TODO: obsłuzyć to
                throw ex;
            }
        }

        private string GetScriptParameters()
        {
            var sb = new StringBuilder();
            foreach (var scriptParameter in _scriptParameters)
            {
                sb.Append($" -e \"{scriptParameter.Key}='{scriptParameter.Value}'\"");
            }
            return sb.ToString();
        }
    }
}
