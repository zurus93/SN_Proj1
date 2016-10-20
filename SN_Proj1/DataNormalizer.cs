using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Resources;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace SN_Proj1
{
    public class DataNormalizer
    {
        private readonly double _normalizedHigh;
        private readonly double _normalizegLow;
        private readonly IDictionary<int, double> _minColumnValues;
        private readonly IDictionary<int, double> _maxColumnValues;



        public DataNormalizer(double[][] data, double normalizedHigh, double normalizedLow)
        {
            _normalizedHigh = normalizedHigh;
            _normalizegLow = normalizedLow;
            _minColumnValues = new Dictionary<int, double>();
            _maxColumnValues = new Dictionary<int, double>();

            PrepareData(data);
        }



        private void PrepareData(double[][] data)
        {
            var vectorSize = data[0].Length; // zakładam, że wszystkie są takie same

            for (var i = 0; i < vectorSize; i++)
            {
                var column = data.Select(x => x[i]).ToArray();
                _minColumnValues[i] = column.Min();
                _maxColumnValues[i] = column.Max();
            }
        }


        public double[][] Normalize(double[][] data, params int[] columns)
        {
            if (columns.Length == 0)
            {
                return data.Select(row => row.Select(Normalize).ToArray()).ToArray();
            }

            List<double[]> result = new List<double[]>();

            foreach (var row in data)
            {
                var rowList = new List<double>();
                var rowItemIndex = 0;
                for (; rowItemIndex < row.Count() && rowItemIndex < columns.Length; rowItemIndex++)
                {
                    rowList.Add(Normalize(row[rowItemIndex], columns[rowItemIndex]));
                }
                for (; rowItemIndex < row.Count(); rowItemIndex++)
                {
                    rowList.Add(row[rowItemIndex]);
                }
                result.Add(rowList.ToArray());
            }

            return result.ToArray();
        }

        public double[] Denormalize(double[] input, params int[] columns)
        {
            var result = new List<double>();
            var rowItemIndex = 0;
            for (; rowItemIndex < input.Count() && rowItemIndex < columns.Length; rowItemIndex++)
            {
                result.Add(Denormalize(input[rowItemIndex], columns[rowItemIndex]));
            }
            for (; rowItemIndex < input.Count(); rowItemIndex++)
            {
                result.Add(input[rowItemIndex]);
            }
            return result.ToArray();
        }

        public double Normalize(double value, int column)
        {
            return (value - _minColumnValues[column]) * (_normalizedHigh - _normalizegLow)
                   / (_maxColumnValues[column] - _minColumnValues[column]) + _normalizegLow;
        }

        public double Denormalize(double value, int column)
        {
            return ((_minColumnValues[column] - _maxColumnValues[column]) * value - _normalizedHigh * _minColumnValues[column] +
                    _maxColumnValues[column] * _normalizegLow) /
                (_normalizegLow - _normalizedHigh);
        }


    }
}
