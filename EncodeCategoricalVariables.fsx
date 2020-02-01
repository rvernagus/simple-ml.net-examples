#load ".paket/load/netcoreapp3.0/main.group.fsx"
open Microsoft.ML
open Microsoft.ML.Data
open System.Collections.Generic
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

[<CLIMutable>]
type EncodedLabel =
    {
        Label : string
        EncodedLabel : bool
    }


if not <| File.Exists("adult.data") then
    use client = new WebClient()
    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")

let context = new MLContext()

let dataView = context.Data.LoadFromTextFile<AdultData>("adult.data", hasHeader = false, separatorChar = ',')

let labelLookup =
    [|
        KeyValuePair("<=50K", false)
        KeyValuePair("<=50K.", false)
        KeyValuePair(">50K", true)
        KeyValuePair(">50K.", true)
    |]

let encoder = context.Transforms.Conversion.MapValue(inputColumnName = "Label", outputColumnName = "EncodedLabel", keyValuePairs = labelLookup)
let transformer = encoder.Fit(dataView)
let transformedDataView = transformer.Transform(dataView)

let encodedLabels = context.Data.CreateEnumerable<EncodedLabel>(transformedDataView, reuseRowObject = true)
do
    encodedLabels
    |> Seq.take 10
    |> Seq.iter (printfn "%A")
