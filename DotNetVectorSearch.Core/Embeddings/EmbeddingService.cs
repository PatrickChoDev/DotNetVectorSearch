using DotNetVectorSearch.Core.RuntimeProvider;
using Microsoft.ML.Tokenizers;

namespace DotNetVectorSearch.Core.Embeddings;

public abstract class EmbeddingService(IRuntimeProvider runtimeProvider, Tokenizer tokenizer) : IEmbeddingService
{
    protected Tokenizer Tokenizer { get; } = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    protected IRuntimeProvider RuntimeProvider { get; } = runtimeProvider ?? throw new ArgumentNullException(nameof(runtimeProvider));

    protected abstract string ModelName { get; }

    public IReadOnlyList<EncodedToken> Tokenize(string text)
    {
        return Tokenizer.EncodeToTokens(text, out _);
    }

    public abstract Task<IReadOnlyList<float>> GetEmbedding(string text);

    public async Task<IEnumerable<float>> GetEmbeddingAsync(string text)
    {
        var result = await GetEmbedding(text);
        return result;
    }

    public async Task<IEnumerable<IEnumerable<float>>> GetEmbeddingsAsync(IEnumerable<string> texts)
    {
        var tasks = texts.Select(GetEmbeddingAsync);
        return await Task.WhenAll(tasks);
    }

    public IEnumerable<string> TokenizeToStringArray(string text)
    {
        return Tokenize(text).Select(token => token.Value).ToArray();
    }

    public string Normalize(string text)
    {
        return Tokenizer.Normalizer == null ? text : Tokenizer.Normalizer.Normalize(text);
    }
}
