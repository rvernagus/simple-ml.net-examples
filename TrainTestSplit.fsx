#r "Microsoft.ML.Core.dll"
#r "Microsoft.ML.Data.dll"
#r "Microsoft.ML.DataView.dll"
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

let trainDataView, testDataView =
     let split = context.Data.TrainTestSplit(dataView, testFraction = 0.2)
     split.TrainSet, split.TestSet

let trainDataEnum = context.Data.CreateEnumerable<AbaloneData>(trainDataView, reuseRowObject = true)
printfn "Counted %d training rows" <| Seq.length trainDataEnum

let testDataEnum = context.Data.CreateEnumerable<AbaloneData>(testDataView, reuseRowObject = true)
printfn "Counted %d test rows" <| Seq.length testDataEnum
