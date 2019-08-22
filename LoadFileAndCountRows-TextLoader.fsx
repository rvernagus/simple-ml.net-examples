#r "Microsoft.ML.Core.dll"
#r "Microsoft.ML.Data.dll"
#r "Microsoft.ML.DataView.dll"
#r "System.Collections.Immutable"
open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open System.Net


if not <| File.Exists("ablone.data") then
    use client = new WebClient()
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data")

let context = new MLContext()

let loader = context.Data.CreateTextLoader([|
        new TextLoader.Column("Sex", DataKind.String, 0)
        new TextLoader.Column("Length", DataKind.Double, 1)
        new TextLoader.Column("Diameter", DataKind.Double, 2)
        new TextLoader.Column("Height", DataKind.Double, 3)
        new TextLoader.Column("WholeWeight", DataKind.Double, 4)
        new TextLoader.Column("ShuckedWeight", DataKind.Double, 5)
        new TextLoader.Column("VisceraWeight", DataKind.Double, 6)
        new TextLoader.Column("ShellWeight", DataKind.Double, 7)
        new TextLoader.Column("Label", DataKind.Int16, 8)
    |], hasHeader = false, separatorChar = ',')

let dataView = loader.Load("abalone.data")
do
    printfn "Counted %A rows" <| dataView.GetRowCount()

    printfn "Counted %d rows" <| Seq.length (dataView.Preview().RowView)

let cursor = dataView.GetRowCursor(dataView.Schema)
let mutable count = 0
while cursor.MoveNext() do
    count <- count+1
do
    printfn "Counted %d rows" count
