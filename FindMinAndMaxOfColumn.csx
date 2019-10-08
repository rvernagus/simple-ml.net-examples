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


if (!File.Exists("ablone.data"))
{
    using var client = new WebClient();
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data");
}

var context = new MLContext();

var dataView = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader: false, separatorChar: ',');

var rings = dataView.GetColumn<float>("Rings");

Console.WriteLine($"Min of Rings is {rings.Min()}");
Console.WriteLine($"Max of Rings is {rings.Max()}");
