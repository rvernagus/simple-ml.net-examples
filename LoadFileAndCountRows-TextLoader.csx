#! "netcoreapp3.0"
#r "nuget: Microsoft.ML, 1.3.1"
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Net;


if (!File.Exists("abalone.data"))
{
    using var client = new WebClient();
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data");
}

var context = new MLContext();

var loader = context.Data.CreateTextLoader(new []
    {
        new TextLoader.Column("Sex", DataKind.String, 0),
        new TextLoader.Column("Length", DataKind.Double, 1),
        new TextLoader.Column("Diameter", DataKind.Double, 2),
        new TextLoader.Column("Height", DataKind.Double, 3),
        new TextLoader.Column("WholeWeight", DataKind.Double, 4),
        new TextLoader.Column("ShuckedWeight", DataKind.Double, 5),
        new TextLoader.Column("VisceraWeight", DataKind.Double, 6),
        new TextLoader.Column("ShellWeight", DataKind.Double, 7),
        new TextLoader.Column("Label", DataKind.Int16, 8),
    },
    hasHeader: false,
    separatorChar: ','  
);

var dataView = loader.Load("abalone.data");
Console.WriteLine($"Counted {dataView.GetRowCount() ?? 0} rows");
Console.WriteLine($"Counted {dataView.Preview().RowView.Count()} rows");

var cursor = dataView.GetRowCursor(dataView.Schema);
var count = 0;
while (cursor.MoveNext())
{
    count += 1;
}

Console.WriteLine($"Counted {count} rows");
