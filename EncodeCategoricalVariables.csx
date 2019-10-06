#! "netcoreapp3.0"
#r "nuget: Microsoft.ML, 1.3.1"
// #r "System.Collections.Immutable"
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;

class AdultData{
    [LoadColumn(0)]
    public float Age { get; set; }
    [LoadColumn(1)]
    public string WorkClass { get; set; }
    [LoadColumn(2)]
    public float Fnlwgt { get; set; }
    [LoadColumn(3)]
    public float Education { get; set; }
    [LoadColumn(4)]
    public float EducationNum { get; set; }
    [LoadColumn(5)]
    public float MaritalStatus { get; set; }
    [LoadColumn(6)]
    public float Occupation { get; set; }
    [LoadColumn(7)]
    public float Relationship { get; set; }
    [LoadColumn(8)]
    public string Race { get; set; }
    [LoadColumn(9)]
    public string Sex { get; set; }
    [LoadColumn(10)]
    public float CapitalGain { get; set; }
    [LoadColumn(11)]
    public float CapitalLoss { get; set; }
    [LoadColumn(12)]
    public float HoursPerWeek { get; set; }
    [LoadColumn(13)]
    public string NativeCountry { get; set; }
    [LoadColumn(14)]
    [ColumnName("Label")]
    public string Target { get; set; }
}

class LabelComparer
{
    public string Label { get; set; }
    public bool EncodedLabel { get; set; }
    public override string ToString() =>
        $"{Label} => {EncodedLabel}";
}

if (!File.Exists("adult.data"))
{
    using var client = new WebClient();
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data");
}

var context = new MLContext();

var dataView = context.Data.LoadFromTextFile<AdultData>("adult.data", hasHeader: false, separatorChar: ',');

var labelLookup = new[]
{
    new KeyValuePair<string, bool>("<=50K", false),
    new KeyValuePair<string, bool>("<=50K.", false),
    new KeyValuePair<string, bool>(">50K", true),
    new KeyValuePair<string, bool>(">50K.", true),
};

var encoder = context.Transforms.Conversion.MapValue(inputColumnName: "Label", outputColumnName: "EncodedLabel", keyValuePairs: labelLookup);
var transformer = encoder.Fit(dataView);
var transformedDataView = transformer.Transform(dataView);

var encodedLabels = context.Data.CreateEnumerable<LabelComparer>(transformedDataView, reuseRowObject: false);
encodedLabels
    .Take(10)
    .ToList()
    .ForEach(Console.WriteLine);
