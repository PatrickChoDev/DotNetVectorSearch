using Microsoft.ML.OnnxRuntime;
using static System.GC;

namespace DotNetVectorSearch.Core.RuntimeProvider;

public class OnnxRuntimeProvider : IRuntimeProvider
{
    private readonly InferenceSession _inferenceSession;

    public OnnxRuntimeProvider(string modelPath, int intraOpThreads = 10, int interOpThreads = 10)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _inferenceSession = CreateInferenceSession(modelPath, intraOpThreads, interOpThreads);
    }

    public void Dispose()
    {
        _inferenceSession.Dispose();
        SuppressFinalize(this);
    }

    public IReadOnlyCollection<string> GetInputNames()
    {
        return _inferenceSession.InputNames;
    }

    public IReadOnlyCollection<string> GetOutputNames()
    {
        return _inferenceSession.OutputNames;
    }

    public async Task<Dictionary<string, OrtValue>> RunInferenceAsync(
        Dictionary<string, OrtValue> inputs,
        IReadOnlyCollection<string>? outputNames = null,
        RunOptions? runOptions = null)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Count == 0) throw new ArgumentException("Inputs cannot be empty.", nameof(inputs));

        ValidateInputs(inputs);

        if (outputNames != null)
            ValidateOutputs(outputNames);

        runOptions = PrepareRunOptions(runOptions);

        outputNames ??= _inferenceSession.OutputNames;

        var results = await Task.Run(() => _inferenceSession.Run(runOptions, inputs, outputNames));

        if (results.Count != outputNames.Count)
            throw new InvalidOperationException($"Expected {outputNames.Count} outputs, but got {results.Count}.");

        return MapOutputsToNames(outputNames, results);
    }

    private static InferenceSession CreateInferenceSession(string modelPath, int? intraOpThreads, int? interOpThreads)
    {
        var sessionOptions = new SessionOptions();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
        sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        if (intraOpThreads.HasValue) sessionOptions.IntraOpNumThreads = intraOpThreads.Value;
        if (interOpThreads.HasValue) sessionOptions.InterOpNumThreads = interOpThreads.Value;
        sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        sessionOptions.LogId = "OnnxRuntimeProvider";
        return new InferenceSession(modelPath, sessionOptions);
    }

    private void ValidateInputs(Dictionary<string, OrtValue> inputs)
    {
        var validInputNames = _inferenceSession.InputNames.ToArray();
        var providedInputNames = inputs.Keys.ToList();
        var invalidInputNames = providedInputNames.Where(inputName => !validInputNames.Contains(inputName)).ToList();
        var missingInputNames = validInputNames.Where(validName => !providedInputNames.Contains(validName)).ToList();

        if (invalidInputNames.Count == 0 && missingInputNames.Count == 0) return;
        var errorMessages = new List<string>();
        if (invalidInputNames.Count != 0)
            errorMessages.Add($"Invalid input name(s): {string.Join(", ", invalidInputNames)}.");
        if (missingInputNames.Count != 0)
            errorMessages.Add($"Missing required input name(s): {string.Join(", ", missingInputNames)}.");
        errorMessages.Add($"Valid input names are: {string.Join(", ", validInputNames)}.");
        throw new ArgumentException(string.Join(" ", errorMessages), nameof(inputs));
    }

    private void ValidateOutputs(IReadOnlyCollection<string> outputNames)
    {
        var validOutputNames = _inferenceSession.OutputNames.ToArray();
        var providedOutputNames = outputNames.ToList();
        var invalidOutputNames =
            providedOutputNames.Where(outputName => !validOutputNames.Contains(outputName)).ToList();
        var missingOutputNames = validOutputNames.Where(validName => !providedOutputNames.Contains(validName)).ToList();

        if (invalidOutputNames.Count == 0 && !missingOutputNames.Any()) return;
        var errorMessages = new List<string>();
        if (invalidOutputNames.Count != 0)
            errorMessages.Add($"Invalid output name(s): {string.Join(", ", invalidOutputNames)}.");
        if (missingOutputNames.Count != 0)
            errorMessages.Add($"Missing required output name(s): {string.Join(", ", missingOutputNames)}.");
        errorMessages.Add($"Valid output names are: {string.Join(", ", validOutputNames)}.");
        throw new ArgumentException(string.Join(" ", errorMessages), nameof(outputNames));
    }

    private static RunOptions PrepareRunOptions(RunOptions? runOptions)
    {
        return runOptions ?? new RunOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
            LogId = "OnnxRuntimeProviderRun"
        };
    }

    private static Dictionary<string, OrtValue> MapOutputsToNames(IReadOnlyCollection<string> outputNames,
        IDisposableReadOnlyCollection<OrtValue> results)
    {
        var outputDict = new Dictionary<string, OrtValue>();
        var i = 0;
        foreach (var name in outputNames)
        {
            outputDict[name] = results[i];
            i++;
        }

        return outputDict;
    }
}
