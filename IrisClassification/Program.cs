using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

class Program
{
    static void Main()
    {
        var mlContext = new MLContext(seed: 1);
        string dataPath = "Iris.csv";
        string modelPath = "irisModel.zip";
        ITransformer model;

        // Load full dataset (needed for metrics)
        // Creates a IDataView which is memory efficient table
        var data = mlContext.Data.LoadFromTextFile<IrisData>(
            dataPath, separatorChar: ',', hasHeader: true);

        // Split into train/test
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        if (File.Exists(modelPath))
        {
            // Load existing model
            model = mlContext.Model.Load(modelPath, out var schema);
            Console.WriteLine("Loaded existing model.");
        }
        else
        {
            // Build pipeline
            var pipeline = mlContext.Transforms
                .Concatenate("Features",
                    nameof(IrisData.SepalLength),
                    nameof(IrisData.SepalWidth),
                    nameof(IrisData.PetalLength),
                    nameof(IrisData.PetalWidth))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.MulticlassClassification.Trainers
                    .SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train
            model = pipeline.Fit(split.TrainSet);
            //It initializes weights for each feature and class.
            // Iteratively optimizes the loss function (cross-entropy) using stochastic dual coordinate ascent.
            // Updates weights until convergence or max iterations.
            
            Console.WriteLine("Model trained.");

            // Save model
            mlContext.Model.Save(model, split.TrainSet.Schema, modelPath);
            Console.WriteLine("Model saved to disk.");
        }

        // Evaluate metrics on the test set
        var predictions = model.Transform(split.TestSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine("\n--- Model Metrics ---");
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");

        // Prediction engine
        var predictionEngine = mlContext.Model
            .CreatePredictionEngine<IrisData, IrisPrediction>(model);

        var sample = new IrisData
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };

        var result = predictionEngine.Predict(sample);
        Console.WriteLine($"\nPredicted flower for sample: {result.PredictedLabels}");
    }
}
