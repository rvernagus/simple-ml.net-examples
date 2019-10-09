#! "netcoreapp3.0"
#r "nuget: Microsoft.ML, 1.3.1"
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Net;


class AbaloneData
{
    [LoadColumn(0)]
    public string Sex { get; set; }

    [LoadColumn(1)]
    public float Length { get; set; }

    [LoadColumn(2)]
    public float Diameter { get; set; }

    [LoadColumn(3)]
    public float Height { get; set; }

    [LoadColumn(4)]
    public float WholeWeight { get; set; }

    [LoadColumn(5)]
    public float ShuckedWeight { get; set; }

    [LoadColumn(6)]
    public float VisceraWeight { get; set; }

    [LoadColumn(7)]
    public float ShellWeight { get; set; }

    [LoadColumn(8)]
    public float Rings { get; set; }
}

class SexComparer
{
    public string Sex { get; set; }
    public float[] EncodedSex { get; set; }

    public override string ToString() => $"{Sex}; [{string.Join(",", EncodedSex)}]";
}

if (!File.Exists("abalone.data"))
{
    using var client = new WebClient();
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data");
}

var context = new MLContext();

var dataView = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader: false, separatorChar: ',');

var encoder = context.Transforms.Categorical.OneHotEncoding(inputColumnName: "Sex", outputColumnName: "EncodedSex");
var transformer = encoder.Fit(dataView);
var transformedDataView = transformer.Transform(dataView);

var encodedLabels = context.Data.CreateEnumerable<SexComparer>(transformedDataView, reuseRowObject: false);
var rand = new Random();
encodedLabels
    .OrderBy(_ => rand.Next())
    .Take(10)
    .ToList()
    .ForEach(x => Console.WriteLine(x))
