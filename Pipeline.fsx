#r "Microsoft.ML.Core.dll"
#r "Microsoft.ML.Data.dll"
#r "Microsoft.ML.DataView.dll"
#r "Microsoft.ML.Transforms.dll"
#r "System.Collections.Immutable"
open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open System.Net


[<CLIMutable>]
type AbaloneData =
    {
        [<LoadColumn(0)>]
        Sex : string
        [<LoadColumn(1)>]
        Length : float32
        [<LoadColumn(2)>]
        Diameter : float32
        [<LoadColumn(3)>]
        Height : float32
        [<LoadColumn(4)>]
        WholeWeight : float32
        [<LoadColumn(5)>]
        ShuckedWeight : float32
        [<LoadColumn(6)>]
        VisceraWeight : float32
        [<LoadColumn(7)>]
        ShellWeight : float32
        [<LoadColumn(8)>]
        Rings : single
    }


if not <| File.Exists("ablone.data") then
    use client = new WebClient()
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data")

let context = new MLContext()

let dataView = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader = false, separatorChar = ',')

let pipeline =
    EstimatorChain()
        .Append(context.Transforms.Categorical.OneHotEncoding("Sex"))
        .Append(context.Transforms.Concatenate("Features", "Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight"))
        .Append(context.Transforms.NormalizeMinMax(outputColumnName = "FeaturesNorm", inputColumnName = "Features"))

dataView.Preview().RowView
|> Seq.take 5
|> Seq.map (fun row -> row.Values)
|> Seq.iter (printfn "%A")

printfn "---------------------------"

let transformer = pipeline.Fit(dataView)
let transformedDataView = transformer.Transform(dataView)
transformedDataView.Preview().RowView
|> Seq.take 5
|> Seq.map (fun row -> row.Values.[12])
|> Seq.map (fun v -> v.Value :?> VBuffer<single>)
|> Seq.map (fun vec -> vec.DenseValues())
|> Seq.iter (fun vals -> (vals |> Seq.iter (printf "%A "); printfn ""))
