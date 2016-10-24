using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SN_Proj1
{
    public class UnrecognizedArgumentException : ApplicationException
    {
    }

    public class MultipleArgumentAssignmentException : ApplicationException
    {
    }

    public class CommandLineOption
    {
        public char ShortNotation { get; set; }
        public string LongNotation { get; set; }
        public bool ParameterRequired { get; set; }
    }


    /// <summary>
    /// Parser linii poleceń, weryfikuje ją i zwraca zarejestrowane polecenia w wygodnej formie.
    /// </summary>
    public class CommandLineParser
    {
        private List<CommandLineOption> _options;
        private Dictionary<CommandLineOption, string> _settings;

        public CommandLineParser(ICollection<CommandLineOption> options)
        {
            _options = options.ToList();
            _settings = new Dictionary<CommandLineOption, string>();
        }

        public void Parse(string[] arguments)
        {
            for (int parsedArguments = 0; parsedArguments < arguments.Length; ++parsedArguments)
            {
                var argument = arguments[parsedArguments];

                if (argument.StartsWith("-"))
                {
                    var option = _options.FirstOrDefault(t => argument == "-" + t.LongNotation);

                    if (option == null)
                    {
                        throw new UnrecognizedArgumentException();
                    }

                    if (_settings.ContainsKey(option))
                    {
                        throw new MultipleArgumentAssignmentException();
                    }

                    if (option.ParameterRequired && (parsedArguments + 1 >= arguments.Length))
                    {
                        throw new Exception();
                    }

                    string value = null;
                    if (option.ParameterRequired)
                    {
                        value = arguments[parsedArguments + 1];
                        ++parsedArguments;
                    }

                    _settings.Add(option, value);

                }
                else
                {
                    for (int i = 0; i < argument.Length; ++i)
                    {
                        var shortOpt = argument[i];
                        var option = _options.First(t => shortOpt == t.ShortNotation);
                        if (option.ParameterRequired && i != argument.Length - 1)
                            throw new Exception();
                        string value = null;
                        if (option.ParameterRequired)
                        {
                            value = arguments[parsedArguments + 1];
                            ++parsedArguments;
                        }
                        _settings.Add(option, value);
                    }
                }
            }
        }

        public bool TryGet(string key, out string value)
        {
            var option = _options.First(t => t.LongNotation == key);
            if (!_settings.ContainsKey(option))
            {
                value = null;
                return false;
            }

            value = _settings[option];
            return true;
        }
    }

}
