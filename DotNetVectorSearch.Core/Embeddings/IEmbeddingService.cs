using Microsoft.ML.Tokenizers;

namespace DotNetVectorSearch.Core.Embeddings;

public interface IEmbeddingService
{
    /// <summary>
    ///     Tokenizes the input text into a list of tokens.
    /// </summary>
    /// <param name="text">The input text to tokenize.</param>
    /// <returns>A list of tokens extracted from the input text.</returns>
    IReadOnlyList<EncodedToken> Tokenize(string text);

    /// <summary>
    ///     Extracts an embedding vector from the input text.
    /// </summary>
    /// <param name="text">The input text to extract the embedding from.</param>
    /// <returns> A list of doubles representing the embedding vector.</returns>
    Task<IReadOnlyList<float>> GetEmbedding(string text);

    Task<IEnumerable<float>> GetEmbeddingAsync(string text);

    Task<IEnumerable<IEnumerable<float>>> GetEmbeddingsAsync(IEnumerable<string> texts);
}
