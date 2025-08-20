using DotNetVectorSearch.Core.Embeddings;
using Microsoft.Data.Sqlite;
using System.Text.Json;

namespace DotNetVectorSearch.WebAPI.Services;

public interface IVectorSearchService
{
    Task<EmbeddingResult> GetEmbeddingAsync(string text);
    Task<BatchEmbeddingResult> GetBatchEmbeddingsAsync(string[] texts);
    Task<SimilarityResult> CalculateSimilarityAsync(string text1, string text2);
    Task<List<DocumentResult>> GetAllDocumentsAsync();
    Task<SimilaritySearchResult> SearchSimilarDocumentsAsync(string queryText, int topK = 5);
}

public class VectorSearchService : IVectorSearchService
{
    private readonly IEmbeddingService _embeddingService;
    private readonly ILogger<VectorSearchService> _logger;
    private readonly string _databasePath;

    public VectorSearchService(IEmbeddingService embeddingService, ILogger<VectorSearchService> logger, IConfiguration configuration)
    {
        _embeddingService = embeddingService ?? throw new ArgumentNullException(nameof(embeddingService));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _databasePath = Path.Combine(AppContext.BaseDirectory,configuration.GetConnectionString("EmbeddingsDatabase") ?? "embeddings.db");
    }

    public async Task<EmbeddingResult> GetEmbeddingAsync(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty", nameof(text));

        try
        {
            _logger.LogInformation("Generating embedding for text: {Text}", text);
            var embedding = await _embeddingService.GetEmbedding(text);
            
            return new EmbeddingResult
            {
                Text = text,
                Embedding = embedding.ToArray(),
                Dimensions = embedding.Count,
                Success = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to generate embedding for text: {Text}", text);
            return new EmbeddingResult
            {
                Text = text,
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    public async Task<BatchEmbeddingResult> GetBatchEmbeddingsAsync(string[] texts)
    {
        if (texts == null || texts.Length == 0)
            throw new ArgumentException("Texts cannot be null or empty", nameof(texts));

        try
        {
            _logger.LogInformation("Generating embeddings for {TextCount} texts", texts.Length);
            var embeddings = await _embeddingService.GetEmbeddingsAsync(texts);
            
            var results = texts.Zip(embeddings, (text, embedding) => new EmbeddingResult
            {
                Text = text,
                Embedding = embedding.ToArray(),
                Dimensions = embedding.Count(),
                Success = true
            }).ToArray();

            return new BatchEmbeddingResult
            {
                Results = results,
                TotalCount = results.Length,
                Success = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to generate batch embeddings");
            return new BatchEmbeddingResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    public async Task<SimilarityResult> CalculateSimilarityAsync(string text1, string text2)
    {
        if (string.IsNullOrWhiteSpace(text1) || string.IsNullOrWhiteSpace(text2))
            throw new ArgumentException("Both texts must be provided");

        try
        {
            _logger.LogInformation("Calculating similarity between two texts");
            var embedding1 = await _embeddingService.GetEmbedding("query: " + text1);
            var embedding2 = await _embeddingService.GetEmbedding("query: " + text2);
            
            var similarity = CalculateCosineSimilarity(embedding1, embedding2);
            
            return new SimilarityResult
            {
                Text1 = text1,
                Text2 = text2,
                Similarity = similarity,
                Embedding1 = embedding1.ToArray(),
                Embedding2 = embedding2.ToArray(),
                Success = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to calculate similarity between texts");
            return new SimilarityResult
            {
                Text1 = text1,
                Text2 = text2,
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    public async Task<List<DocumentResult>> GetAllDocumentsAsync()
    {
        try
        {
            _logger.LogInformation("Retrieving all documents from database");
            var documents = new List<DocumentResult>();

            _logger.LogDebug("Using database path: {DatabasePath}", _databasePath);
            await using var connection = new SqliteConnection($"Data Source={_databasePath}");
            await connection.OpenAsync();
            
            const string query = "SELECT id, question, answer, combined_text, embedding, embedding_dimensions, created_at FROM documents ORDER BY id";
            await using var command = new SqliteCommand(query, connection);
            await using var reader = await command.ExecuteReaderAsync();
            
            while (await reader.ReadAsync())
            {
                var embeddingJson = reader.GetString(4); // embedding column
                var embedding = JsonSerializer.Deserialize<float[]>(embeddingJson) ?? [];
                
                documents.Add(new DocumentResult
                {
                    Id = reader.GetInt32(0), // id column
                    Question = reader.GetString(1), // question column  
                    Answer = reader.GetString(2), // answer column
                    CombinedText = reader.GetString(3), // combined_text column
                    Embedding = embedding,
                    EmbeddingDimensions = reader.GetInt32(5), // embedding_dimensions column
                    CreatedAt = reader.GetDateTime(6) // created_at column
                });
            }
            
            _logger.LogInformation("Retrieved {DocumentCount} documents from database", documents.Count);
            return documents;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to retrieve documents from database");
            return new List<DocumentResult>();
        }
    }

    public async Task<SimilaritySearchResult> SearchSimilarDocumentsAsync(string queryText, int topK = 5)
    {
        if (string.IsNullOrWhiteSpace(queryText))
            throw new ArgumentException("Query text cannot be null or empty", nameof(queryText));

        try
        {
            _logger.LogInformation("Searching for similar documents to query: {QueryText}", queryText);
            
            // Generate embedding for query text
            var queryEmbedding = await _embeddingService.GetEmbedding("query: " +queryText);
            
            // Get all documents from database
            var documents = await GetAllDocumentsAsync();
            
            // Calculate similarities and rank
            var similarities = documents.Select(doc => new DocumentSimilarity
            {
                Document = doc,
                Similarity = CalculateCosineSimilarity(queryEmbedding, doc.Embedding)
            })
            .OrderByDescending(x => x.Similarity)
            .Take(topK)
            .ToList();
            
            return new SimilaritySearchResult
            {
                QueryText = queryText,
                QueryEmbedding = queryEmbedding.ToArray(),
                Results = similarities,
                TotalDocuments = documents.Count,
                Success = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to search for similar documents");
            return new SimilaritySearchResult
            {
                QueryText = queryText,
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    private static float CalculateCosineSimilarity(IReadOnlyList<float> vector1, IReadOnlyList<float> vector2)
    {
        if (vector1.Count != vector2.Count)
            throw new ArgumentException("Vectors must have the same dimensions");

        var dotProduct = 0f;
        var magnitude1 = 0f;
        var magnitude2 = 0f;

        for (var i = 0; i < vector1.Count; i++)
        {
            dotProduct += vector1[i] * vector2[i];
            magnitude1 += vector1[i] * vector1[i];
            magnitude2 += vector2[i] * vector2[i];
        }

        magnitude1 = (float)Math.Sqrt(magnitude1);
        magnitude2 = (float)Math.Sqrt(magnitude2);

        if (magnitude1 == 0f || magnitude2 == 0f)
            return 0f;

        return dotProduct / (magnitude1 * magnitude2);
    }
}

// Data Transfer Objects
public class EmbeddingResult
{
    public string Text { get; init; } = string.Empty;
    public float[] Embedding { get; init; } = [];
    public int Dimensions { get; init; }
    public bool Success { get; init; }
    public string? ErrorMessage { get; init; }
}

public class BatchEmbeddingResult
{
    public EmbeddingResult[] Results { get; init; } = [];
    public int TotalCount { get; set; }
    public bool Success { get; init; }
    public string? ErrorMessage { get; init; }
}

public class SimilarityResult
{
    public string Text1 { get; init; } = string.Empty;
    public string Text2 { get; init; } = string.Empty;
    public float Similarity { get; init; }
    public float[] Embedding1 { get; init; } = [];
    public float[] Embedding2 { get; init; } = [];
    public bool Success { get; init; }
    public string? ErrorMessage { get; init; }
}

public class DocumentResult
{
    public int Id { get; init; }
    public string Question { get; init; } = string.Empty;
    public string Answer { get; init; } = string.Empty;
    public string CombinedText { get; init; } = string.Empty;
    public float[] Embedding { get; init; } = [];
    public int EmbeddingDimensions { get; init; }
    public DateTime CreatedAt { get; init; }
}

public class DocumentSimilarity
{
    public DocumentResult Document { get; init; } = new();
    public float Similarity { get; init; }
}

public class SimilaritySearchResult
{
    public string QueryText { get; init; } = string.Empty;
    public float[] QueryEmbedding { get; init; } = [];
    public List<DocumentSimilarity> Results { get; init; } = [];
    public int TotalDocuments { get; init; }
    public bool Success { get; init; }
    public string? ErrorMessage { get; init; }
}
