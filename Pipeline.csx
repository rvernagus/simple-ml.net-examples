#! "netcoreapp3.0"
#r "nuget: Microsoft.ML, 1.3.1"
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
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

if (!File.Exists("abalone.data"))
{
    using var client = new WebClient();
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data");
}

var context = new MLContext();

var dataView = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader: false, separatorChar: ',');
dataView = context.Data.ShuffleRows(dataView);

var pipeline = context.Transforms
    .Categorical.OneHotEncoding("Sex")
    .Append(context.Transforms.Concatenate("Features", "Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight"))
    .Append(context.Transforms.NormalizeMinMax(outputColumnName: "FeaturesNorm", inputColumnName: "Features"));

var transformer = pipeline.Fit(dataView);
var transformedDataView = transformer.Transform(dataView);

var features = transformedDataView.GetColumn<float[]>("Features");
var featuresNorm = transformedDataView.GetColumn<float[]>("FeaturesNorm");


Console.WriteLine("Unprocessed Data\n----------------------");
var items = dataView.Preview().RowView
    .Take(5)
    .Select(x => string.Join(", ", x.Values));
foreach (var item in items)
{
    Console.WriteLine(item);
}

Console.WriteLine("One-Hot Encoded Data\n----------------------");
items = features
    .Take(5)
    .Select(x => string.Join(", ", x));
foreach (var item in items)
{
    Console.WriteLine(item);
}

Console.WriteLine("One-Hot Encoded and Normalized Data\n----------------------");
items = featuresNorm
    .Take(5)
    .Select(x => string.Join(", ", x));
foreach (var item in items)
{
    Console.WriteLine(item);
}
