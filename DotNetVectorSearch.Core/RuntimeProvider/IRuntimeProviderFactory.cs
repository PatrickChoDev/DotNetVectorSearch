namespace DotNetVectorSearch.Core.RuntimeProvider;

public interface IRuntimeProviderFactory
{
    Task<IRuntimeProvider> CreateAsync(string key, CancellationToken cancellationToken = default);
}
