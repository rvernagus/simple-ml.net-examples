#r "Microsoft.ML.Core.dll"
#r "Microsoft.ML.Data.dll"
#r "Microsoft.ML.DataView.dll"
#r "Microsoft.ML.Transforms.dll"
#r "System.Collections.Immutable"
open Microsoft.ML
open Microsoft.ML.Data
open System


[<CLIMutable>]
type SourceData =
    {
        Date: DateTime
    }

[<CLIMutable>]
type MappedData =
    {
        mutable SourceDate: DateTime

        mutable Month: int

        mutable IsWeekend: bool
    }


let context = new MLContext()

let data =
    [
        { Date = DateTime.Parse("2019-08-20") }
        { Date = DateTime.Parse("2019-07-06") }
        { Date = DateTime.Parse("2019-06-16") }
        { Date = DateTime.Parse("2019-05-14") }
    ]

let dataView = context.Data.LoadFromEnumerable<SourceData>(data)

let mapper source mapped =
    mapped.SourceDate <- source.Date
    mapped.Month <- source.Date.Month
    mapped.IsWeekend <- source.Date.DayOfWeek = DayOfWeek.Saturday || source.Date.DayOfWeek = DayOfWeek.Sunday
let mapperAction = new Action<SourceData, MappedData>(mapper)

let encoder = context.Transforms.CustomMapping(mapperAction, "MapDate")
let transformer = encoder.Fit(dataView)
let transformedDataView = transformer.Transform(dataView)

let mappedData = context.Data.CreateEnumerable<MappedData>(transformedDataView, reuseRowObject = true)
do
    mappedData
    |> Seq.iter (printfn "%A")
