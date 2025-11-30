using Microsoft.ML.Data;

public class IrisPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabels { get; set; } = "";
}