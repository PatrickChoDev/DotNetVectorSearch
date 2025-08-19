using DotNetVectorSearch.Core.Embeddings;
using DotNetVectorSearch.WebAPI.Services;
using DotNetVectorSearch.WebAPI.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.OpenApi.Models;
using System.Reflection;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "Vector Search API",
        Version = "v1",
        Description = "A comprehensive API for text embeddings and semantic similarity search using E5 Multilingual model",
        Contact = new OpenApiContact
        {
            Name = "PatrickChoDev",
            Email = "devpatrick.cho@gmail.com",
            Url = new Uri("https://github.com/PatrickChoDev")
        }
    });

    // Include XML comments for better Swagger documentation
    var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
    var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);
    if (File.Exists(xmlPath))
    {
        c.IncludeXmlComments(xmlPath);
    }

    // Add examples and descriptions
    c.EnableAnnotations();
});

// Register services
builder.Services.AddSingleton<IEmbeddingService>(provider =>
{
    var logger = provider.GetRequiredService<ILogger<E5MultilingualEmbeddings>>();
    return new E5MultilingualEmbeddings(logger);
});

builder.Services.AddScoped<IVectorSearchService, VectorSearchService>();

// Add CORS for all environments
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAny", policy =>
        policy.AllowAnyOrigin()
               .AllowAnyMethod()
               .AllowAnyHeader());
});

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "Vector Search API v1");
        c.RoutePrefix = string.Empty; // Set Swagger UI at apps root
        c.DocumentTitle = "Vector Search API Documentation";
        c.DisplayRequestDuration();
    });
}

// Enable CORS for all environments
app.UseCors("AllowAny");

app.UseHttpsRedirection();

// Health check endpoint
app.MapGet("/health", () => Results.Ok(new { Status = "Healthy", Timestamp = DateTime.UtcNow }))
    .WithName("HealthCheck")
    .WithTags("Health")
    .WithSummary("Health check endpoint")
    .WithDescription("Returns the health status of the API");

// Single text embedding endpoint
app.MapPost("/api/embeddings", async (
    [FromBody] EmbeddingRequest request,
    [FromServices] IVectorSearchService vectorSearchService) =>
{
    try
    {
        var result = await vectorSearchService.GetEmbeddingAsync(request.Text);
        
        if (!result.Success)
        {
            return Results.BadRequest(new ApiResponse<object>
            {
                Success = false,
                ErrorMessage = result.ErrorMessage
            });
        }

        return Results.Ok(new ApiResponse<EmbeddingResponse>
        {
            Success = true,
            Data = new EmbeddingResponse
            {
                Text = result.Text,
                Embedding = result.Embedding,
                Dimensions = result.Dimensions
            }
        });
    }
    catch (Exception ex)
    {
        return Results.Problem($"Internal server error: {ex.Message}");
    }
})
.WithName("GetEmbedding")
.WithTags("Embeddings")
.WithSummary("Generate embedding for a single text")
.WithDescription("Generates a 384-dimensional embedding vector for the provided text using E5 Multilingual model");

// Batch embeddings endpoint
app.MapPost("/api/embeddings/batch", async (
    [FromBody] BatchEmbeddingRequest request,
    [FromServices] IVectorSearchService vectorSearchService) =>
{
    try
    {
        var result = await vectorSearchService.GetBatchEmbeddingsAsync(request.Texts);
        
        if (!result.Success)
        {
            return Results.BadRequest(new ApiResponse<object>
            {
                Success = false,
                ErrorMessage = result.ErrorMessage
            });
        }

        var responses = result.Results.Select(r => new EmbeddingResponse
        {
            Text = r.Text,
            Embedding = r.Embedding,
            Dimensions = r.Dimensions
        }).ToArray();

        return Results.Ok(new ApiResponse<EmbeddingResponse[]>
        {
            Success = true,
            Data = responses
        });
    }
    catch (Exception ex)
    {
        return Results.Problem($"Internal server error: {ex.Message}");
    }
})
.WithName("GetBatchEmbeddings")
.WithTags("Embeddings")
.WithSummary("Generate embeddings for multiple texts")
.WithDescription("Generates 384-dimensional embedding vectors for multiple texts in a single request");

// Similarity calculation endpoint
app.MapPost("/api/similarity", async (
    [FromBody] SimilarityRequest request,
    [FromServices] IVectorSearchService vectorSearchService) =>
{
    try
    {
        var result = await vectorSearchService.CalculateSimilarityAsync(request.Text1, request.Text2);
        
        if (!result.Success)
        {
            return Results.BadRequest(new ApiResponse<object>
            {
                Success = false,
                ErrorMessage = result.ErrorMessage
            });
        }

        var response = new SimilarityResponse
        {
            Text1 = result.Text1,
            Text2 = result.Text2,
            Similarity = result.Similarity
        };

        // Include embeddings if requested
        if (request.IncludeEmbeddings)
        {
            response.Embedding1 = result.Embedding1;
            response.Embedding2 = result.Embedding2;
        }

        return Results.Ok(new ApiResponse<SimilarityResponse>
        {
            Success = true,
            Data = response
        });
    }
    catch (Exception ex)
    {
        return Results.Problem($"Internal server error: {ex.Message}");
    }
})
.WithName("CalculateSimilarity")
.WithTags("Similarity")
.WithSummary("Calculate similarity between two texts")
.WithDescription("Calculates cosine similarity between two texts. Returns a score from 0 (completely different) to 1 (identical)");

// Get all documents endpoint
app.MapGet("/api/documents", async (
    [FromServices] IVectorSearchService vectorSearchService,
    [FromQuery] bool includeEmbeddings = false) =>
{
    try
    {
        var documents = await vectorSearchService.GetAllDocumentsAsync();
        
        var responses = documents.Select(doc => new DocumentResponse
        {
            Id = doc.Id,
            Question = doc.Question,
            Answer = doc.Answer,
            CombinedText = doc.CombinedText,
            EmbeddingDimensions = doc.EmbeddingDimensions,
            CreatedAt = doc.CreatedAt,
            Embedding = includeEmbeddings ? doc.Embedding : null
        }).ToList();

        return Results.Ok(new ApiResponse<List<DocumentResponse>>
        {
            Success = true,
            Data = responses
        });
    }
    catch (Exception ex)
    {
        return Results.Problem($"Internal server error: {ex.Message}");
    }
})
.WithName("GetAllDocuments")
.WithTags("Documents")
.WithSummary("Get all documents from the database")
.WithDescription("Retrieves all question-answer pairs from the embedded documents database");

// Similarity search endpoint
app.MapPost("/api/search", async (
    [FromBody] SimilaritySearchRequest request,
    [FromServices] IVectorSearchService vectorSearchService) =>
{
    try
    {
        var result = await vectorSearchService.SearchSimilarDocumentsAsync(request.QueryText, request.TopK);
        
        if (!result.Success)
        {
            return Results.BadRequest(new ApiResponse<object>
            {
                Success = false,
                ErrorMessage = result.ErrorMessage
            });
        }

        var response = new SimilaritySearchResponse
        {
            QueryText = result.QueryText,
            QueryEmbedding = request.IncludeEmbeddings ? result.QueryEmbedding : null,
            TotalDocuments = result.TotalDocuments,
            ResultCount = result.Results.Count,
            Results = result.Results.Select(r => new SimilarDocumentResponse
            {
                Document = new DocumentResponse
                {
                    Id = r.Document.Id,
                    Question = r.Document.Question,
                    Answer = r.Document.Answer,
                    CombinedText = r.Document.CombinedText,
                    EmbeddingDimensions = r.Document.EmbeddingDimensions,
                    CreatedAt = r.Document.CreatedAt,
                    Embedding = request.IncludeEmbeddings ? r.Document.Embedding : null
                },
                Similarity = r.Similarity
            }).ToList()
        };

        return Results.Ok(new ApiResponse<SimilaritySearchResponse>
        {
            Success = true,
            Data = response
        });
    }
    catch (Exception ex)
    {
        return Results.Problem($"Internal server error: {ex.Message}");
    }
})
.WithName("SearchSimilarDocuments")
.WithTags("Search")
.WithSummary("Search for similar documents")
.WithDescription("Finds the most similar documents to a query text using semantic similarity search");

app.Run();
