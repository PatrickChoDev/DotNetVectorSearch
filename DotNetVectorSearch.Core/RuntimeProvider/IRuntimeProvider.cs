using Microsoft.ML.OnnxRuntime;

namespace DotNetVectorSearch.Core.RuntimeProvider;

public interface IRuntimeProvider : IDisposable
{
    /// <summary>
    ///     Runs inference on the model with the provided inputs.
    /// </summary>
    /// <param name="inputs">Dictionary mapping input names to OrtValue or Tensor</param>
    /// <param name="outputNames">Optional: list of output names to fetch. If null, returns all outputs.</param>
    /// <param name="runOptions">Optional: run options for the inference session.</param>
    /// <returns>Dictionary of output name to OrtValue</returns>
    Task<Dictionary<string, OrtValue>> RunInferenceAsync(
        Dictionary<string, OrtValue> inputs,
        IReadOnlyCollection<string>? outputNames = null,
        RunOptions? runOptions = null);

    IReadOnlyCollection<string> GetInputNames();
    IReadOnlyCollection<string> GetOutputNames();
}
