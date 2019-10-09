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


if not <| File.Exists("abalone.data") then
    use client = new WebClient()
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data")

let context = new MLContext()

let dataView = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader = false, separatorChar = ',')

let pipeline =
    EstimatorChain()
        .Append(context.Transforms.Categorical.OneHotEncoding("Sex"))
        .Append(context.Transforms.Concatenate("Features", "Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight"))
        .Append(context.Transforms.NormalizeMinMax(outputColumnName = "FeaturesNorm", inputColumnName = "Features"))

let transformer = pipeline.Fit(dataView)
let transformedDataView = transformer.Transform(dataView)

let features = transformedDataView.GetColumn<single[]>("Features")
let featuresNorm = transformedDataView.GetColumn<single[]>("FeaturesNorm")

do
    featuresNorm
    |> Seq.zip features
    |> Seq.map (fun (x, y) -> Seq.zip x y)
    |> Seq.head
    |> Seq.iter (printfn "%A")
