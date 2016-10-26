using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;
using Encog.Util.CSV;
using System.Globalization;

namespace SN_Proj1
{
    public class CsvHelper
    {
        public static double[][] Read(string path)
        {
            var reader = new ReadCSV(path, true, CSVFormat.DecimalPoint);

            IList<double[]> data = new List<double[]>();

            while (reader.Next())
            {
                data.Add(ReadLine(reader));
            }
            return data.ToArray();
        }


        private static double[] ReadLine(ReadCSV csv)
        {
            var line = new double[csv.GetCount()];
            for (int i = 0; i < csv.GetCount(); i++)
            {
                line[i] = csv.GetDouble(i);
            }
            return line;
        }

        public static void Write(string path, double[][] input, double[][] output)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentNullException(nameof(path));
            }
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }
            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }
            if (input.Length != output.Length)
            {
                throw new ArgumentException();
            }


            var content = new StringBuilder();
            var rowCount = Math.Max(input.Length, output.Length);


            for (int i = 0; i < rowCount; i++)
            {
                var line = input[i].ToList().Concat(output[i]);

                content.AppendLine(string.Join(",", line.Select(x => x.ToString(CultureInfo.InvariantCulture)).ToArray()));
            }

            System.IO.File.WriteAllText(path, content.ToString());
        }



        public static void Write(string path, double[][] input)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new ArgumentNullException(nameof(path));
            }
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            var content = new StringBuilder();
            var rowCount = input.Length;


            for (int i = 0; i < rowCount; i++)
            {
                content.AppendLine(string.Join(",", input[i].Select(x => x.ToString(CultureInfo.InvariantCulture)).ToArray()));
            }

            System.IO.File.WriteAllText(path, content.ToString());
        }
    }
}
