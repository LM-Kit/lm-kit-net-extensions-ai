using LMKit.Abstractions;
using Microsoft.Extensions.AI;

namespace LMKit.Integrations.ExtensionsAI.Embeddings
{
    /// <summary>
    /// Convenience extensions for bridging Microsoft.Extensions.AI embedding generators into LMKit.
    /// </summary>
    public static class EmbeddingGeneratorExtensions
    {
        /// <summary>
        /// Wraps a Microsoft.Extensions.AI <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> as
        /// an LMKit <see cref="IEmbedder"/>, ready to pass to <c>RagEngine</c> or any other consumer.
        /// </summary>
        /// <param name="generator">The embedding generator to adapt. Cannot be null.</param>
        /// <param name="modelId">An optional model identifier; inferred from the generator when null.</param>
        /// <param name="embeddingSize">The vector dimension when known; learned from the first call when 0.</param>
        /// <returns>An <see cref="IEmbedder"/> backed by <paramref name="generator"/>.</returns>
        public static IEmbedder AsEmbedder(
            this IEmbeddingGenerator<string, Embedding<float>> generator,
            string? modelId = null,
            int embeddingSize = 0)
        {
            return new ExtensionsAIEmbedder(generator, modelId, embeddingSize);
        }
    }
}
