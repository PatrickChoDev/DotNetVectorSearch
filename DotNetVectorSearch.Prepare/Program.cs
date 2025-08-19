using DotNetVectorSearch.Core.Embeddings;
using Microsoft.Extensions.Logging;
using Microsoft.Data.Sqlite;
using System.Text.Json;

// Create logger
using var loggerFactory = LoggerFactory.Create(builder =>
    builder.AddConsole().SetMinimumLevel(LogLevel.Information));
var logger = loggerFactory.CreateLogger<Program>();

try
{
    // Initialize E5 Multilingual Embeddings
    var embeddingService = new E5MultilingualEmbeddings(logger);
    
    // Initialize SQLite database - save in project root directory
    var projectRoot = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(AppContext.BaseDirectory)))!;
    var dbPath = Path.Combine(projectRoot, "embeddings.db");
    await InitializeDatabaseAsync(dbPath, logger);
    
    // Read the CSV dataset from the current project directory
    var csvPath = Path.Combine(projectRoot, "dataset.csv");
    if (!File.Exists(csvPath))
    {
        logger.LogError("Dataset file not found at {CsvPath}", csvPath);
        return;
    }

    var lines = await File.ReadAllLinesAsync(csvPath);
    logger.LogInformation("Processing {LineCount} lines from dataset", lines.Length);

    // Process and store embeddings
    await using var connection = new SqliteConnection($"Data Source={dbPath}");
    await connection.OpenAsync();
    
    var insertCount = 0;

    // Skip the header row and process each entry
    for (var i = 1; i < lines.Length; i++)
    {
        var parts = ParseCsvLine(lines[i]);
        if (parts.Length >= 3)
        {
            var id = int.Parse(parts[0]);
            var question = parts[1];
            var answer = parts[2];

            logger.LogInformation("Processing entry {Id}: {Question}", id, question);

            // Combine question and answer for single embedding
            var combinedText = $"{question} : {answer}";
            var embedding = await embeddingService.GetEmbedding("passage: " + combinedText);

            // Store a single document in a database
            await StoreDocumentAsync(connection, id, question, answer, combinedText, embedding.ToArray(), logger);

            insertCount++;

            logger.LogInformation("Generated and stored document embedding - {Dimensions} dimensions for combined Q&A", 
                embedding.Count);
        }
    }

    logger.LogInformation("Dataset processing completed successfully. Stored {InsertCount} document embeddings in database", insertCount);
    logger.LogInformation("Database saved as: {DbPath}", Path.GetFullPath(dbPath));
}
catch (Exception ex)
{
    logger.LogError(ex, "An error occurred while processing the dataset");
}

return;

static async Task InitializeDatabaseAsync(string dbPath, ILogger logger)
{
    logger.LogInformation("Initializing SQLite database at {DbPath}", dbPath);
    
    // Delete existing database if it exists
    if (File.Exists(dbPath))
    {
        File.Delete(dbPath);
        logger.LogInformation("Deleted existing database file");
    }

    await using var connection = new SqliteConnection($"Data Source={dbPath}");
    await connection.OpenAsync();

    const string createTableSql = """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            combined_text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            embedding_dimensions INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_documents_id ON documents(id);
        CREATE INDEX idx_documents_created_at ON documents(created_at);
        CREATE INDEX idx_documents_question ON documents(question);
        """;

    await using var command = new SqliteCommand(createTableSql, connection);
    await command.ExecuteNonQueryAsync();
    
    logger.LogInformation("Database schema created successfully");
}

static async Task StoreDocumentAsync(SqliteConnection connection, int id, string question, string answer, string combinedText, float[] embedding, ILogger logger)
{
    const string insertSql = """
        INSERT INTO documents (id, question, answer, combined_text, embedding, embedding_dimensions)
        VALUES (@id, @question, @answer, @combined_text, @embedding, @embedding_dimensions)
        """;

    await using var command = new SqliteCommand(insertSql, connection);
    command.Parameters.AddWithValue("@id", id);
    command.Parameters.AddWithValue("@question", question);
    command.Parameters.AddWithValue("@answer", answer);
    command.Parameters.AddWithValue("@combined_text", combinedText);
    command.Parameters.AddWithValue("@embedding", JsonSerializer.Serialize(embedding));
    command.Parameters.AddWithValue("@embedding_dimensions", embedding.Length);

    await command.ExecuteNonQueryAsync();
    
    logger.LogDebug("Stored document for ID {Id} ({Dimensions} dimensions)", 
        id, embedding.Length);
}

static string[] ParseCsvLine(string line)
{
    var result = new List<string>();
    var current = "";
    var inQuotes = false;
    
    foreach (var c in line)
    {
        switch (c)
        {
            case '"':
                inQuotes = !inQuotes;
                break;
            case ',' when !inQuotes:
                result.Add(current);
                current = "";
                break;
            default:
                current += c;
                break;
        }
    }
    
    result.Add(current);
    return result.Select(s => s.Trim('"')).ToArray();
}
