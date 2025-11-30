
using Microsoft.ML.Data;
// LoadColumn delar upp och mappar varje kolumn i en fil till en variabel. Växtens attribut i detta fall.
public class IrisData
{
    [LoadColumn(0)] public float Id { get; set; }     
    [LoadColumn(1)] public float SepalLength { get; set; }
    [LoadColumn(2)] public float SepalWidth { get; set; }
    [LoadColumn(3)] public float PetalLength { get; set; }
    [LoadColumn(4)] public float PetalWidth { get; set; }
    [LoadColumn(5)] public string Label { get; set; }   
}