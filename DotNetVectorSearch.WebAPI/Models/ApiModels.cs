using System.ComponentModel.DataAnnotations;

namespace DotNetVectorSearch.WebAPI.Models;

// Request Models
public class EmbeddingRequest
{
    /// <summary>
    /// The text to generate an embedding for
    /// </summary>
    /// <example>How do I cancel my hotel booking?</example>
    [Required]
    [MinLength(1)]
    public string Text { get; set; } = string.Empty;
}

public class BatchEmbeddingRequest
{
    /// <summary>
    /// Array of texts to generate embeddings for
    /// </summary>
    /// <example>["How do I cancel my booking?", "What is the refund policy?"]</example>
    [Required]
    [MinLength(1)]
    public string[] Texts { get; set; } = Array.Empty<string>();
}

public class SimilarityRequest
{
    /// <summary>
    /// First text for similarity comparison
    /// </summary>
    /// <example>How do I cancel my hotel booking?</example>
    [Required]
    [MinLength(1)]
    public string Text1 { get; set; } = string.Empty;

    /// <summary>
    /// Second text for similarity comparison
    /// </summary>
    /// <example>Can I cancel my reservation?</example>
    [Required]
    [MinLength(1)]
    public string Text2 { get; set; } = string.Empty;

    /// <summary>
    /// Whether to include the embedding vectors in the response
    /// </summary>
    /// <example>false</example>
    public bool IncludeEmbeddings { get; set; } = false;
}

public class SimilaritySearchRequest
{
    /// <summary>
    /// Query text to find similar documents for
    /// </summary>
    /// <example>How to cancel booking?</example>
    [Required]
    [MinLength(1)]
    public string QueryText { get; set; } = string.Empty;

    /// <summary>
    /// Number of top similar documents to return
    /// </summary>
    /// <example>5</example>
    [Range(1, 50)]
    public int TopK { get; set; } = 5;

    /// <summary>
    /// Whether to include embedding vectors in the response
    /// </summary>
    /// <example>false</example>
    public bool IncludeEmbeddings { get; set; } = false;
}

// Response Models
public class ApiResponse<T>
{
    /// <summary>
    /// Indicates if the operation was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// The response data
    /// </summary>
    public T? Data { get; set; }

    /// <summary>
    /// Error message if the operation failed
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Timestamp of the response
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class EmbeddingResponse
{
    /// <summary>
    /// The original input text
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// The generated embedding vector
    /// </summary>
    public float[] Embedding { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Number of dimensions in the embedding
    /// </summary>
    public int Dimensions { get; set; }
}

public class SimilarityResponse
{
    /// <summary>
    /// First input text
    /// </summary>
    public string Text1 { get; set; } = string.Empty;

    /// <summary>
    /// Second input text
    /// </summary>
    public string Text2 { get; set; } = string.Empty;

    /// <summary>
    /// Cosine similarity score between the two texts (0-1, where 1 is identical)
    /// </summary>
    public float Similarity { get; set; }

    /// <summary>
    /// Embedding vector for the first text (optional)
    /// </summary>
    public float[]? Embedding1 { get; set; }

    /// <summary>
    /// Embedding vector for the second text (optional)
    /// </summary>
    public float[]? Embedding2 { get; set; }
}

public class DocumentResponse
{
    /// <summary>
    /// Document ID from the dataset
    /// </summary>
    public int Id { get; set; }

    /// <summary>
    /// The question text
    /// </summary>
    public string Question { get; set; } = string.Empty;

    /// <summary>
    /// The answer text
    /// </summary>
    public string Answer { get; set; } = string.Empty;

    /// <summary>
    /// Combined question and answer text used for embedding
    /// </summary>
    public string CombinedText { get; set; } = string.Empty;

    /// <summary>
    /// Embedding vector dimensions
    /// </summary>
    public int EmbeddingDimensions { get; set; }

    /// <summary>
    /// When the document was created
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// The embedding vector (optional, for performance reasons)
    /// </summary>
    public float[]? Embedding { get; set; }
}

public class SimilarDocumentResponse
{
    /// <summary>
    /// The document information
    /// </summary>
    public DocumentResponse Document { get; set; } = new();

    /// <summary>
    /// Similarity score to the query (0-1, where 1 is most similar)
    /// </summary>
    public float Similarity { get; set; }
}

public class SimilaritySearchResponse
{
    /// <summary>
    /// The original query text
    /// </summary>
    public string QueryText { get; set; } = string.Empty;

    /// <summary>
    /// Query embedding vector (optional)
    /// </summary>
    public float[]? QueryEmbedding { get; set; }

    /// <summary>
    /// List of similar documents ranked by similarity
    /// </summary>
    public List<SimilarDocumentResponse> Results { get; set; } = new();

    /// <summary>
    /// Total number of documents in the database
    /// </summary>
    public int TotalDocuments { get; set; }

    /// <summary>
    /// Number of results returned
    /// </summary>
    public int ResultCount { get; set; }
}
