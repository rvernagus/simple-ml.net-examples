open System.Collections.Generic

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
type AdultData =
    {
        [<LoadColumn(0)>]
        Age : float32
        [<LoadColumn(1)>]
        WorkClass : string
        [<LoadColumn(2)>]
        Fnlwgt : float32
        [<LoadColumn(3)>]
        Education : string
        [<LoadColumn(4)>]
        EducationNum : float32
        [<LoadColumn(5)>]
        MaritalStatus : float32
        [<LoadColumn(6)>]
        Occupation : float32
        [<LoadColumn(7)>]
        Relationship : float32
        [<LoadColumn(8)>]
        Race : string
        [<LoadColumn(9)>]
        Sex : string
        [<LoadColumn(10)>]
        CapitalGain : float32
        [<LoadColumn(11)>]
        CapitalLoss : float32
        [<LoadColumn(12)>]
        HoursPerWeek : float32
        [<LoadColumn(13)>]
        NativeCountry : string
        [<LoadColumn(14)>]
        [<ColumnName("Label")>]
        Target : string
    }


if not <| File.Exists("adult.data") then
    use client = new WebClient()
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")

let context = new MLContext()

let dataView = context.Data.LoadFromTextFile<AdultData>("adult.data", hasHeader = false, separatorChar = ',')

dataView.Preview().RowView
|> Seq.take 5
|> Seq.map (fun row -> row.Values.[14])
|> Seq.iter (printfn "%A")

printfn "---------------------------"

let labelLookup =
    [|
        KeyValuePair("<=50K", false)
        KeyValuePair("<=50K.", false)
        KeyValuePair(">50K", true)
        KeyValuePair(">50K.", true)
    |]

let encoder = context.Transforms.Conversion.MapValue("Label", labelLookup)
let transformer = encoder.Fit(dataView)
let transformedDataView = transformer.Transform(dataView)

transformedDataView.Preview().RowView
|> Seq.take 5
|> Seq.map (fun row -> [row.Values.[14]; row.Values.[15]])
|> Seq.iter (printfn "%A")
