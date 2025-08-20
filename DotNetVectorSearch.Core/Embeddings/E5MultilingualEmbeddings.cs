using DotNetVectorSearch.Core.RuntimeProvider;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;

namespace DotNetVectorSearch.Core.Embeddings;

public class E5MultilingualEmbeddings : EmbeddingService
{
    private const int MaxLength = 512;
    private readonly ILogger _logger;

    public E5MultilingualEmbeddings(ILogger logger)
        : base(
            CreateRuntimeProvider(logger),
            CreateTokenizer(logger))
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _logger.LogInformation("E5MultilingualEmbeddings initialized successfully");
    }

    protected override string ModelName => nameof(E5MultilingualEmbeddings);

    private static OnnxRuntimeProvider CreateRuntimeProvider(ILogger logger)
    {
        // Get the path to the solution root directory and then to the Core project's Onnx folder
        var baseDirectory = AppContext.BaseDirectory;
        var coreOnnxPath = Path.Combine(baseDirectory ?? throw new InvalidOperationException("Could not determine solution root path"), "Onnx");
        var modelPath = Path.Combine(coreOnnxPath, "model_O4.onnx");
        
        if (!File.Exists(modelPath))
        {
            logger.LogError("Model file for e5-multilingual not found at {ModelPath}", modelPath);
            throw new InvalidOperationException($"Model file for e5-multilingual not found at {modelPath}.");
        }

        logger.LogInformation("Creating ONNX runtime provider with model at {ModelPath}", modelPath);
        return new OnnxRuntimeProvider(modelPath, 20, 40);
    }

    private static SentencePieceTokenizer CreateTokenizer(ILogger logger)
    {
        // Get the path to the solution root directory and then to the Core project's Onnx folder
        var baseDirectory = AppContext.BaseDirectory;
        var coreOnnxPath = Path.Combine(baseDirectory ?? throw new InvalidOperationException("Could not determine solution root path"), "Onnx");
        var filePath = Path.Combine(coreOnnxPath, "sentencepiece.bpe.model");

        if (!File.Exists(filePath))
        {
            logger.LogError("Tokenizer file for e5-multilingual not found at {FilePath}", filePath);
            throw new InvalidOperationException($"Tokenizer file for e5-multilingual not found at {filePath}.");
        }

        logger.LogInformation("Creating tokenizer from file at {FilePath}", filePath);

        try
        {
            // Use FileStream directly instead of loading the entire file into memory
            using var fileStream = File.OpenRead(filePath);
            var tokenizer = SentencePieceTokenizer.Create(fileStream, true, true);

            if (tokenizer == null)
            {
                logger.LogError("Failed to create tokenizer from file at {FilePath}", filePath);
                throw new InvalidOperationException($"Failed to create tokenizer from file at {filePath}.");
            }

            logger.LogInformation("Tokenizer created successfully");
            return tokenizer;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Exception occurred while creating tokenizer from {FilePath}", filePath);
            throw;
        }
    }

    public override async Task<IReadOnlyList<float>> GetEmbedding(string text)
    {
        if (string.IsNullOrEmpty(text))
            throw new ArgumentException("Text cannot be null or empty", nameof(text));

        try
        {
            var ids = ProcessTokens(text);
            var inputs = CreateModelInputs(ids);
            var outputs = await RuntimeProvider.RunInferenceAsync(inputs);

            return ProcessEmbeddings(outputs);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to generate embedding for text: {Text}", text);
            throw;
        }
    }

    private long[] ProcessTokens(string text)
    {
        var encoded = Tokenizer.EncodeToTokens(text,out _);
        var tokens = encoded.ToList();
        var ids = tokens.Select((token, idx) =>
        {
            if (token.Value is "<s>" or "</s>")
                // Special tokens are typically assigned specific IDs
                return token.Value == "<s>" && idx == 0
                    ? 0L // Start token (CLS token)
                    : token.Id;

            return token.Id + 1L;
        }).ToArray();

        // Handle sequence length (E5 models typically use max 512 tokens)
        if (ids.Length <= MaxLength) return ids;
        ids = ids.Take(MaxLength).ToArray();
        _logger.LogDebug("Truncated sequence to {MaxLength} tokens", MaxLength);

        return ids;
    }

    private static Dictionary<string, OrtValue> CreateModelInputs(long[] ids)
    {
        var seqLength = ids.Length;
        var idsShape = new long[] { 1, seqLength }; // [batch_size, sequence_length]
        var memoryInfo = OrtMemoryInfo.DefaultInstance;

        // Create input_ids tensor (int64)
        var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory<long>(memoryInfo, ids, idsShape);

        // Create attention_mask tensor (all ones for real tokens)
        var attentionMask = Enumerable.Repeat(1L, seqLength).ToArray();
        var attentionMaskOrtValue = OrtValue.CreateTensorValueFromMemory<long>(memoryInfo, attentionMask, idsShape);

        // Create token_type_ids tensor (all zeros for single sequence)
        var tokenTypeIds = new long[seqLength]; // defaults to 0
        var tokenTypeIdsOrtValue = OrtValue.CreateTensorValueFromMemory<long>(memoryInfo, tokenTypeIds, idsShape);

        return new Dictionary<string, OrtValue>
        {
            { "input_ids", inputIdsOrtValue },
            { "attention_mask", attentionMaskOrtValue },
            { "token_type_ids", tokenTypeIdsOrtValue }
        };
    }

    private static float[] ProcessEmbeddings(IReadOnlyDictionary<string, OrtValue> outputs)
    {
        var lastHiddenState = outputs["last_hidden_state"].GetTensorDataAsSpan<float>();
        var shape = outputs["last_hidden_state"].GetTensorTypeAndShape().Shape;

        if (shape.Length != 3)
            throw new InvalidOperationException("Unexpected shape for last_hidden_state: " +
                                                string.Join(", ", shape));

        var hiddenSize = (int)shape[2];

        // Use CLS token pooling (first token) - return raw embeddings
        var clsEmbedding = ExtractClsEmbedding(lastHiddenState, hiddenSize);

        return NormalizeEmbedding(clsEmbedding);
    }

    private static float[] ExtractClsEmbedding(ReadOnlySpan<float> lastHiddenState, int hiddenSize)
    {
        var clsEmbedding = new float[hiddenSize];
        for (var j = 0; j < hiddenSize; j++)
            clsEmbedding[j] = lastHiddenState[j]; // First token (CLS token) at index 0

        return clsEmbedding;
    }

    private static float[] NormalizeEmbedding(float[] clsEmbedding)
    {
        var magnitude = Math.Sqrt(clsEmbedding.Sum(x => x * x));
        var normalizedEmbedding = new float[clsEmbedding.Length];

        if (magnitude > 1e-12) // Avoid division by zero
        {
            for (var j = 0; j < clsEmbedding.Length; j++) normalizedEmbedding[j] = (float)(clsEmbedding[j] / magnitude);
        }
        else
        {
            Array.Copy(clsEmbedding, normalizedEmbedding, clsEmbedding.Length);
        }

        return normalizedEmbedding;
    }
}
