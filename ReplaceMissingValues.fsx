#r "Microsoft.ML.Core.dll"
#r "Microsoft.ML.Data.dll"
#r "Microsoft.ML.DataView.dll"
#r "Microsoft.ML.Transforms.dll"
#r "System.Collections.Immutable"
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms
open System.IO
open System.Net


let makeSingleColumn index =
    new TextLoader.Column(string index, DataKind.Single, index)


if not <| File.Exists("arrhythmia.data") then
    use client = new WebClient()
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data", "arrhythmia.data")

let context = new MLContext()

let columns =
    seq { 0..278 }
    |> Seq.map makeSingleColumn
    |> Seq.append [ new TextLoader.Column("Label", DataKind.Int32, 279) ]
    |> Seq.toArray

let featureColumns =
    seq { 0..278 }
    |> Seq.map string
    |> Seq.toArray

let textLoader = context.Data.CreateTextLoader(columns, hasHeader = false, separatorChar = ',')

let allDataView = textLoader.Load("arrhythmia.data")

allDataView.Preview().RowView
|> Seq.take 5
|> Seq.map (fun row -> row.Values.[14])
|> Seq.iter (printfn "%A")

printfn "---------------------------"

let transformer =
    EstimatorChain()
    |> fun chain -> chain.Append(context.Transforms.Concatenate("Features", featureColumns))
    |> fun chain -> chain.Append(context.Transforms.ReplaceMissingValues("Features", replacementMode = MissingValueReplacingEstimator.ReplacementMode.Mean))
    |> (fun pipeline -> pipeline.Fit(allDataView))


let transformedDataView = transformer.Transform(allDataView)

transformedDataView.Preview().RowView
|> Seq.take 5
|> Seq.map (fun row -> row.Values.[281])
|> Seq.map (fun v -> v.Value :?> VBuffer<single>)
|> Seq.map (fun vec -> vec.DenseValues())
|> Seq.iter (fun vals -> printfn "%A" (Seq.item 13 vals))
