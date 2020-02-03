#! "netcoreapp3.0"
#r "nuget: Microsoft.ML, 1.3.1"
using Microsoft.ML;
using System;
using System.Linq;

class SourceData
{
    public DateTime Date { get; set; }
}

class MappedData
{
    public DateTime SourceDate { get; set; }
    public int Month { get; set; }
    public bool IsWeekend { get; set; }

    public override string ToString() =>
        $"{SourceDate, 25} {Month, 5} {IsWeekend, 9}";
}

var context = new MLContext();

var data = new[]
{
    new SourceData { Date = DateTime.Parse("2019-08-20") },
    new SourceData { Date = DateTime.Parse("2019-07-06") },
    new SourceData { Date = DateTime.Parse("2019-06-16") },
    new SourceData { Date = DateTime.Parse("2019-05-14") },
};

var dataView = context.Data.LoadFromEnumerable<SourceData>(data);

var encoder = context.Transforms.CustomMapping<SourceData, MappedData>((input, output) =>
{
    output.SourceDate = input.Date;
    output.Month = input.Date.Month;
    output.IsWeekend = input.Date.DayOfWeek == DayOfWeek.Saturday || input.Date.DayOfWeek == DayOfWeek.Sunday;
}, "MapDate");

var transformer = encoder.Fit(dataView);
var transformedDataView = transformer.Transform(dataView);

Console.WriteLine($"{"SourceDate", 25} {"Month", 5} {"IsWeekend", 9}");

var mappedData = context.Data.CreateEnumerable<MappedData>(transformedDataView, reuseRowObject: false);
foreach (var item in mappedData)
{
    Console.WriteLine(item);
}
